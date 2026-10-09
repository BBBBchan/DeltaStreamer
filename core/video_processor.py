import os
import argparse
from typing import List, Literal, Dict, Any, Set, Union
import random
import glob
from pathlib import Path
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import re
import json

from swift.infer_engine import RequestConfig
from swift import InferStats

# Image-tokenisation budget. Defaults target the Qwen3.5/3.6 dense models
# (~100k px/frame). A run script may override any of these via env BEFORE
# import (e.g. the Qwen3-VL run script raises MAX_PIXELS) — hence setdefault
# rather than unconditional assignment, so the override is honoured.
os.environ.setdefault('MIN_PIXELS', '3136')
os.environ.setdefault('MAX_PIXELS', '100352')
os.environ.setdefault('INPUT_SIZE', '448')
os.environ.setdefault('MAX_NUM', '1')

VIDEO_PREFIX = ""

STOP_ON_RESPONSE_TASKS = {"EPM", "HLD", "ASI", "STU", "OJR", "ATR", "FPD", "OCR", "ACR", "CRR", "SSR"}

BACKWARD_TASKS = ["EPM", "ASI", "HLD"]
REALTIME_TASKS = ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]
FORWARD_TASKS = ["REC", "SSR", "CRR"]

# One unified answer-format constraint shared by every benchmark / task.
# We never special-case task type in the prompt — we just tell the model:
#   "answer with the format the question asks for; if there are options,
#    pick one of them; do not refuse to answer."
GENERIC_ANSWER_CONSTRAINT = (
    "Answer with the format the question asks for. "
    "If the question lists explicit options, your final answer MUST be "
    "one of those listed options (reply with the option letter followed "
    "by its full text). If the question expects Yes/No, reply exactly "
    "with 'Yes' or 'No'. Do NOT output text outside the requested answer "
    "space. If you are uncertain, give your best-guess answer rather than "
    "refusing."
)

# REC prompt mode: "letter" (A-K MCQ -> 0..10) or "number" (free-form integer 0..10).
REC_PROMPT_MODE = "letter"

def _normalize_task_filter(task_filter: Union[None, Set[str], List[str], str]) -> Set[str]:
    if task_filter is None:
        return set()
    if isinstance(task_filter, str):
        if task_filter.strip().lower() in {"", "*", "all", "__all__"}:
            return set()
        task_filter = {t.strip() for t in task_filter.split(",") if t.strip()}
    elif isinstance(task_filter, (list, set)):
        task_filter = {str(t).strip() for t in task_filter if str(t).strip()}
    else:
        raise ValueError(
            "task_filter must be None, str, list or set, got: %s" % type(task_filter)
        )
    return task_filter

def load_dataset_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {json_path}")
    
    processed_samples = []
    for sample in data:
        is_event_state_schema = bool(
            sample.get("all_steps")
            and not sample.get("question")
            and not sample.get("activity")
        )
        if is_event_state_schema:
            # SSR 数据有两种来源：
            # (1) 原 JSON 已经把每个 step 单独展开成一行（含 int
            #     `step_start_time` 和单字符串 `all_steps`，常带
            #     `disable_ssr_expand: True`）；
            # (2) 旧版聚合 JSON（all_steps 为 \n 分隔多个 step、
            #     start_time/start_times 为 list）。
            #
            # 历史 BUG：sample.get("start_time", []) 拼写错误（实际
            # 字段是 step_start_time/start_times），且即使 case-(1)
            # 已含 step_start_time 也会被覆盖为 0，导致所有 SSR 样本
            # question_time 全部被强制成 0，模型在 t=0 必答 No。
            #
            # 修复：
            #   * 若已带 disable_ssr_expand=True 且 all_steps 是单
            #     字符串，直接保留原样本（不展开），保留原
            #     step_start_time。
            #   * 否则（旧版聚合）按 \n 拆分 all_steps，并优先取
            #     `start_times`/`step_start_time` list。
            if sample.get("disable_ssr_expand") and isinstance(
                sample.get("all_steps"), str
            ) and "\n" not in sample["all_steps"]:
                # 单 step 样本：原 step_start_time 已在 sample 里，
                # 直接透传，仅补 step_index 字段。
                new_sample = sample.copy()
                new_sample.setdefault("step_index", 0)
                new_sample.setdefault(
                    "original_id", sample.get("id"))
                new_sample["id"] = sample.get("id")  # 保持原 id
                processed_samples.append(new_sample)
                continue

            all_steps = sample["all_steps"]
            # 优先 start_times（复数），其次 step_start_time（如果是 list），
            # 否则 fallback 到 start_time（旧字段，可能是逗号分隔字符串）。
            start_times = sample.get("start_times")
            if start_times is None:
                sst_field = sample.get("step_start_time")
                if isinstance(sst_field, list):
                    start_times = sst_field
                else:
                    start_times = sample.get("start_time", [])

            if isinstance(all_steps, str):
                steps = [step.strip() for step in all_steps.split('\n') if step.strip()]
            elif isinstance(all_steps, list):
                steps = all_steps
            else:
                processed_samples.append(sample)
                continue

            if isinstance(start_times, str):
                start_times = [int(t.strip()) for t in start_times.split(',') if t.strip()]
            elif not isinstance(start_times, list):
                start_times = []

            for i, step in enumerate(steps):
                new_sample = sample.copy()
                new_sample["all_steps"] = step
                new_sample["step_index"] = i
                new_sample["step_start_time"] = (
                    int(start_times[i]) if i < len(start_times) else
                    # fallback: 若原样本已含 int step_start_time，沿用
                    int(sample["step_start_time"])
                    if isinstance(sample.get("step_start_time"), int)
                    else 0
                )
                new_sample["original_id"] = sample["id"]
                new_sample["id"] = f"{sample['id']}_step_{i}"
                processed_samples.append(new_sample)
        else:
            processed_samples.append(sample)
    
    print(f"After processing SSR samples: {len(processed_samples)} samples")
    return processed_samples

def load_processed_ids_from_jsonl(jsonl_path: str) -> set:
    processed = set()
    if not jsonl_path or not os.path.exists(jsonl_path):
        return processed
    cnt = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sid = obj.get("id") or obj.get("sample_id")
                if sid:
                    processed.add(sid)
                    cnt += 1
            except Exception as e:
                print(f"Warning: skip bad line in {jsonl_path}: {e}")
    print(f"Loaded {cnt} lines, found {len(processed)} processed ids from {jsonl_path}")
    return processed

SYSTEM = """
You are a helpful assistant specializing in streaming video analysis. 
You will receive input frame by frame, each labeled with absolute time intervals 
in the exact format <Xs-Ys> (e.g., <0s-1s>). Follow these rules precisely:

1. Use </Silence> when:
   - No relevant event has started, OR
   - The current input is irrelevant to the given question.  

2. Use </Standby> when:
   - An event is in progress but has not yet completed, OR
   - The current input is relevant but the question cannot yet be answered.  

3. Use </Response> only when:
   - An event has fully concluded, OR
   - The available information is sufficient to fully answer the question.
   Provide a complete description at this point.  

Do not provide partial answers or speculate beyond the given information.  
Whenever you deliver an answer, begin with </Response>.
"""

class VideoProcessor:
    def __init__(self, sample: dict, system: str):
        self.sample = sample
        video_path = sample["video"]
        # Prepared datasets store valid frame-directory paths.
        self.sample["video"] = video_path  # update in-place for downstream use
        self.video_path = os.path.join(VIDEO_PREFIX, video_path)
        self.system = system
        self.current_round = 0
        self.data = None
        self.output = {}
        self.completed = False
        
        image_files = sorted(os.listdir(self.video_path))
        # One round corresponds to 1 second (matches `<Xs-Ys>` prompt template
        # and the `realtime`/`question_time` unit in the dataset). Do NOT
        # subsample by 2 here, otherwise EPM-style tasks where
        # question_time == realtime can never trigger inference.
        self.frame_paths = [os.path.join(self.video_path, f) for f in image_files]
        stream_end_time = sample.get("stream_end_time")
        if stream_end_time is not None:
            stream_end_time = int(stream_end_time)
            if stream_end_time < 1:
                raise ValueError(
                    f"stream_end_time must be positive: {stream_end_time}"
                )
            self.frame_paths = self.frame_paths[:stream_end_time]
        self.total_rounds = len(self.frame_paths)
        
        self.question, self.question_time = self._build_question()
        # Fallback: 若 question_time 落在最后一帧之后（含等于
        # total_rounds 的"边界值"），把它 clamp 到 total_rounds-1，让
        # 模型在视频最后一帧 *正常推理*——这是 last frame 抛题、调
        # 工具、commit `</Response>` 的常规路径。COMMIT round（虚拟
        # current_round=total_rounds）只作为 last-frame 没赶上 commit
        # 的 *兜底* 救命轮，不应被默认用作"首次抛题"的位置（COMMIT
        # round 工具被禁，没法做证据收集）。
        #
        # 实测 OVO-Bench 数据：HLD/OJR/STU/ATR/CRR 等任务的字段值
        # （realtime/ask_time）经常恰好等于 total_rounds，触发此
        # clamp 是正常预期行为，不会"丢失"任何信息——模型仍可在 last
        # frame 看到题目并答题。
        if self.total_rounds > 0 and self.question_time >= self.total_rounds:
            self.question_time = self.total_rounds - 1
        
    def get_next_request(self, last_answer=None) -> tuple:
        if self.current_round == 0:
            self.data = get_data_stream_window(
                frame_paths=self.frame_paths,
                round_num=self.current_round,
                system=self.system,
                question=self.question,
                question_time=self.question_time
            )
        else:
            self.data = get_data_stream_window(
                frame_paths=self.frame_paths,
                round_num=self.current_round,
                system=self.system,
                question=self.question,
                data=self.data,
                answer=last_answer,
                question_time=self.question_time
            )
        
        should_process = self.current_round >= self.question_time
        
        if should_process:
            return True, InferRequest(**self.data)
        else:
            return False, None

    def _build_question(self) -> tuple:
        task = self.sample["task"]
        if task in {"EPM", "HLD", "STU", "OJR", "ATR", "FPD", "OCR", "ACR"}:
            tmpl = (
                "{question}\n{options}\n"
                + GENERIC_ANSWER_CONSTRAINT
            )
            question_time = self.sample["realtime"]
            options = self.sample["options"]
            opts = "\n".join([f"{chr(65 + i)}. {o}" for i, o in enumerate(options)])
            return tmpl.format(question=self.sample["question"], options=opts), int(question_time)
        elif task in {"ASI"}:
            tmpl = (
                "{question}\n{options}\n"
                + GENERIC_ANSWER_CONSTRAINT
            )
            # OVO-Bench ASI 原始 JSON 已提供 `ask_time` 字段（中位数恰好
            # 在视频中段，realtime 的 50%），这是论文评测协议指定的
            # 提问时刻——通常落在 trigger 动作刚完成的瞬间，模型需要
            # 据此回答 "after X 做了什么 / before X 做了什么"。
            #
            # 历史 BUG：之前用 random.randint(0, realtime)，引入两个
            # 问题：(1) 跑次不可复现（每次 seed 不同，accuracy 波动）；
            # (2) random 经常落在 trigger 动作前，模型无 MEMORY 可查
            # 也无法等待，注定答错（同一份 subset 在 06-05 vs 06-09
            # 两次跑相差 +25pt 即此 random 方差引起）。
            #
            # 修复：优先取 ask_time；缺失时回退到旧 random 公式。
            at_raw = self.sample.get('ask_time')
            if at_raw is None:
                at_raw = random.randint(0, int(self.sample["realtime"]))
            question_time = int(at_raw)
            options = self.sample["options"]
            opts = "\n".join([f"{chr(65 + i)}. {o}" for i, o in enumerate(options)])
            return tmpl.format(question=self.sample["question"], options=opts), int(question_time)
        elif task == "CRR":
            # ====== CRR 元问题协议（关键）======
            # CRR 是 *元问题*：你不需要给出原问题的答案，只需判断
            # "截至当前为止，原问题所需的关键证据是否已经在视频里出现"。
            # 反例（常见错误）：模型在 reasoning 时答原问题，最终因
            # "动作还没完成 / 还没到达" 输出 "No"。这是错配——只要
            # 关键事件已出现，就应该输出 "Yes"，无需事件全部完成。
            tmpl = (
                "*** META-QUESTION (CRR). DO NOT answer the question "
                "below; ONLY judge whether sufficient evidence has "
                "already appeared in the video to answer it. ***\n"
                "Original question (FOR REFERENCE — do NOT answer it): "
                "{question}\n\n"
                "Your task: based ONLY on the video content shown so "
                "far (up to the current frame), has the key evidence "
                "needed to answer the original question already "
                "appeared in the video?\n"
                "  - 'Yes' = the relevant event / clue / action has "
                "BEGUN OR OCCURRED in the video so far. You do NOT "
                "need the event to be fully completed; the moment the "
                "evidence is observable, the answer is 'Yes'.\n"
                "  - 'No' = the relevant event / clue has NOT yet "
                "happened or is not yet observable in any past frame "
                "(MEMORY contains no matching observation).\n"
                "Heuristics:\n"
                "  * If MEMORY (or the current frame) contains an "
                "observation that directly addresses the original "
                "question's TARGET (e.g. an actor / object / action / "
                "OCR text mentioned in the question), answer 'Yes'.\n"
                "  * If you find yourself reasoning about WHAT the "
                "answer to the original question is, you have "
                "misunderstood the task — STOP and re-judge: 'Has the "
                "evidence appeared? yes or no'.\n"
                "  * 'Walking towards X', 'opening Y', 'starting Z' "
                "all count as evidence appearing — the action does "
                "NOT have to be complete.\n"
                "  * **BEGUN-heuristic via state tags** (改动 6): "
                "scan MEMORY's `actions:` fields for state tags "
                "matching the question's keywords. If you find ANY "
                "tag of `[STARTING]` / `[IN-PROGRESS]` / `[FINISHED]` "
                "for the relevant subject/verb, answer 'Yes' "
                "(evidence has appeared). Only answer 'No' when "
                "MEMORY is genuinely empty for those keywords AND the "
                "video so far is only setup/preamble (title cards, "
                "empty scene, irrelevant b-roll).\n"
                "Your final answer MUST be exactly one word: 'Yes' or 'No'."
            )
            return tmpl.format(question=self.sample["question"]), self.sample["ask_time"]
        elif task == "SSR":
            # ====== SSR 元问题协议（关键）======
            # SSR 是 *元问题*：你不需要描述事件本身，只需判断
            # "截至当前帧，这个事件是否已经在视频中完成（哪怕完成于
            # 视频更早的时刻）"。
            # 反例（常见错误）：模型默认 'No'，因为它在 step_start_time
            # 这一刻刚被问，"还没看见完成动作就回答"。这是错配——
            # 应当扫描 MEMORY 中的所有过去观察：只要曾经看到事件
            # 完成过，就回答 'Yes'。
            question = self.sample["all_steps"]
            tmpl = (
                "*** META-QUESTION (SSR). DO NOT describe or explain "
                "the event below; ONLY judge whether it has ALREADY "
                "HAPPENED (started/occurred) at any point in the video "
                "shown so far. ***\n"
                "Event: '{question}'\n\n"
                "Your task: based ONLY on the video content shown so "
                "far (current frame plus all past observations in "
                "MEMORY), has this event ALREADY OCCURRED — i.e. has "
                "the action taken place or clearly begun and progressed "
                "at some earlier or current moment?\n"
                "  - 'Yes' = the event has happened or is visibly "
                "underway (you saw the action itself, OR a "
                "SEMANTICALLY-EQUIVALENT action, OR the workflow has "
                "already moved PAST this step to a later one).\n"
                "  - 'No' = the event has NOT happened yet — nothing in "
                "MEMORY matches it and the workflow has not reached it.\n"
                "★ KEY judging rules (read carefully):\n"
                "  * Do NOT insist on a literal wording match or an "
                "explicit `[FINISHED]` tag. Real observations describe "
                "actions in GENERIC terms (e.g. 'weaving strands at the "
                "crown with a metal tool' IS 'putting on hair "
                "extensions'; 'spreading adhesive' IS 'apply glue'). "
                "Judge by MEANING, not exact words.\n"
                "  * If MEMORY shows a semantically-matching action that "
                "has been IN-PROGRESS for a few seconds or longer, treat "
                "the event as HAVING OCCURRED → answer 'Yes'. An event "
                "that is actively happening HAS happened.\n"
                "  * TEMPORAL / workflow cue: these events are steps in "
                "an ordered procedure. If MEMORY shows a LATER step is "
                "already underway (or the scene has clearly advanced "
                "beyond this step), then this earlier step must have "
                "already occurred → 'Yes'.\n"
                "  * Only answer 'No' when the matching action is truly "
                "ABSENT from MEMORY, or is only just about to start with "
                "no progress yet.\n"
                "  * Do NOT default to 'No' merely because you did not "
                "see a crisp 'completion' moment — most real actions "
                "have no clean end frame.\n"
                "Your final answer MUST be exactly one word: 'Yes' or 'No'."
            )
            question_time = self.sample.get("step_start_time", 0)
            return tmpl.format(question=question), int(question_time)
        elif task == "REC":
            # OVO-Bench REC 原始 JSON 已提供 `query_time` 字段，标记了
            # "视频末尾的累计计数提问时刻"——这才是真实评测协议（视频被
            # 切到 query_time 处，模型在视频结束时回答总计数）。
            #
            # 历史 BUG：之前用 max(start_times[0]-3, 0) 计算，导致提问
            # 比 query_time 提前 3 秒，恰好落在 activity 第一次出现 *前*
            # （因为 OVO 数据集所有样本恒满足 query_time ==
            # start_times[0]，且 start_times 数组只截取首次出现）。模型
            # 在 activity 还没出现时就被强迫答题，几乎只能输出 "A. 0"，
            # 这正是 REC 准确率长期偏低的核心根因。
            #
            # 修复：优先取 query_time；只有当字段缺失时才回退到旧公式。
            qt_raw = self.sample.get('query_time')
            if qt_raw is None:
                start_times = self.sample.get('start_times') or []
                qt_raw = (start_times[0] if start_times else 0)
            question_time = int(qt_raw)
            if REC_PROMPT_MODE == "number":
                tmpl = (
                    "How many times did activity '{activity}' occur?\n"
                    "Reply with only a single integer between 0 and 10 (inclusive). "
                    "Update your answer if it becomes different at a later time."
                )
                return tmpl.format(activity=self.sample["activity"]), int(question_time)
            tmpl = (
                "How many times did activity '{activity}' occur?\n{options}\n"
                + GENERIC_ANSWER_CONSTRAINT
            )

            # GT range is "0"..."10" (11 values). Map A->0, B->1, ..., K->10.
            options = '\n'.join(f"{chr(65 + i)}. {i}" for i in range(11))

            return tmpl.format(activity=self.sample["activity"], options=options), int(question_time)
        else:
            return "", 0

    def update(self, answer: str):
        self.output[f"Round {self.current_round}"] = answer
        # Strip <think>...</think> reasoning blocks before scanning for the
        # `</Response>` sentinel. Qwen3.5 with thinking enabled often quotes
        # the literal token `</Response>` inside its chain-of-thought (e.g.
        # "I should NOT use </Response> yet"). Without stripping, that quoted
        # token would prematurely terminate the sample.
        _resp_part = _strip_thinking(answer)
        if self.sample.get("task") in STOP_ON_RESPONSE_TASKS and answer and "</Response>" in _resp_part:
            self.current_round += 1
            self.completed = True
            return

        self.current_round += 1
        if self.current_round >= self.total_rounds:
            self.completed = True

    def is_completed(self) -> bool:
        return self.completed


# Match a complete <think>...</think> block (multi-line, non-greedy). We also
# tolerate an unterminated trailing <think> block at the end of the string,
# which can happen when the model is mid-reasoning and the response was cut
# off by max_tokens — in that case we drop everything after the opening tag.
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_TAIL_RE = re.compile(r"<think\b[^>]*>.*\Z", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove any <think>...</think> reasoning span from ``text``.

    Used by both the streaming `update()` loop and the post-hoc
    `_extract_final_response()` so that literal occurrences of
    ``</Response>`` / ``</Silence>`` inside the model's chain-of-thought
    are NEVER confused with the actual response sentinel.
    """
    if not text:
        return text or ""
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _THINK_OPEN_TAIL_RE.sub("", cleaned)
    return cleaned


def _extract_final_response(output_dict: Dict[str, str]):
    """Extract the last </Response> text and the round it was triggered at.

    Returns (final_text, trigger_round). If no Response triggered, returns (None, None).
    """
    def _scan(canonical_only: bool):
        final_text = None
        trigger_round = None
        for key, value in output_dict.items():
            if not value:
                continue
            match = re.fullmatch(r"Round\s+(\d+)", str(key))
            if canonical_only and match is None:
                continue
            value_eff = _strip_thinking(value)
            if "</Response>" not in value_eff:
                continue
            round_index = int(match.group(1)) if match is not None else None
            text = value_eff.split("</Response>", 1)[1].strip()
            final_text = text
            trigger_round = round_index
        return final_text, trigger_round

    # Suffixed round keys preserve planner/audit attempts for trajectory
    # diagnosis. They are not final predictions and must never override the
    # canonical ``Round N`` value merely because they were inserted later.
    canonical = _scan(canonical_only=True)
    return canonical if canonical[0] is not None else _scan(
        canonical_only=False
    )


def _extract_option_letter(response: str):
    """Extract A-K from a response string (A-D for MCQ tasks, A-K for REC mapping 0-10)."""
    if response is None:
        return None
    resp = response.strip()
    if not resp:
        return None
    # Single letter
    if len(resp) == 1 and resp.upper() in "ABCDEFGHIJK":
        return resp.upper()
    # Letter + punctuation: "A.", "B) ", "C: "
    m = re.match(r"\s*([A-Ka-k])\s*[.)\s:：\-]", resp)
    if m:
        return m.group(1).upper()
    # "answer is X" / "option X" / "choose X"
    m = re.search(r"(?:answer|option|choose|select|choice)\s+(?:is\s+)?([A-Ka-k])\b",
                  resp, re.I)
    if m:
        return m.group(1).upper()
    # Fallback: last standalone letter
    letters = re.findall(r"\b([A-Ka-k])\b", resp)
    if letters:
        return letters[-1].upper()
    return None


def _score_one(record: dict) -> int:
    """Score a single record according to OVO-Bench task semantics.

    - BACKWARD/REALTIME (EPM/ASI/HLD/STU/OJR/ATR/ACR/OCR/FPD): letter == gt
    - REC: digit-extracted answer (or letter->index) == gt
    - SSR/CRR: triggered </Response> -> "Yes"; else -> "No". Compare to gt.
      (Mirrors the official semantics: each streamo sample is already split
      to a single (step, realtime) chunk truncated to the test_info time,
      so whether the model emits a Response on that chunk == Yes/No answer.)
    """
    task = record.get("task")
    gt = record.get("gt")
    final_text = record.get("final_response")
    trigger_round = record.get("trigger_round")

    if task in BACKWARD_TASKS or task in REALTIME_TASKS:
        if final_text is None:
            return 0
        letter = _extract_option_letter(final_text)
        if letter is None:
            # Substring fallback (mirrors official offline scoring)
            return int(isinstance(gt, str) and gt.upper() in final_text.upper())
        return int(isinstance(gt, str) and letter == gt.upper())

    if task == "REC":
        if final_text is None:
            return 0
        if REC_PROMPT_MODE == "number":
            # Free-form integer mode: extract first digit run (clamped to 0..10).
            m = re.search(r"\d+", final_text)
            if m:
                try:
                    pred = int(m.group(0))
                    return int(pred == int(gt))
                except Exception:
                    return 0
            return 0
        # REC prompt uses A-K letters mapping to 0..10 (11 values).
        # Prefer letter extraction; fall back to digit extraction.
        letter = _extract_option_letter(final_text)
        if letter is not None:
            pred_num = ord(letter) - ord("A")  # A->0, B->1, ..., K->10
            try:
                return int(pred_num == int(gt))
            except Exception:
                return 0
        # Fallback: any digit run
        digits = "".join(re.findall(r"\d+", final_text))
        if digits:
            try:
                return int(int(digits) == int(gt))
            except Exception:
                return 0
        return 0

    if task in ("SSR", "CRR"):
        # Streamo split truncates each sample to test_info realtime; the model
        # is now required to explicitly answer "Yes" or "No". Try to extract
        # an explicit Yes/No from the response; fall back to "non-empty == Yes"
        # to retain compatibility with older results.
        if final_text is None:
            pred = "No"
        else:
            txt = final_text.strip()
            m = re.search(r"\b(yes|no)\b", txt, re.I)
            if m:
                pred = m.group(1).capitalize()
            else:
                pred = "Yes"
        return int(isinstance(gt, str) and pred == gt)

    return 0


def score_results(jsonl_path: str) -> dict:
    """Read jsonl and compute per-task / per-category / total accuracy.

    Returns a summary dict and prints a formatted report.
    """
    if not os.path.exists(jsonl_path):
        print(f"[score] file not found: {jsonl_path}")
        return {}

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue

    per_task: Dict[str, List[int]] = {}
    for r in records:
        # Backfill final_response if missing (older results pre-fix).
        if "final_response" not in r:
            ft, tr = _extract_final_response(r.get("output", {}))
            r["final_response"] = ft
            r["trigger_round"] = tr
        s = _score_one(r)
        r["score"] = s
        per_task.setdefault(r.get("task", "UNK"), []).append(s)

    def _avg(lst):
        return 100 * sum(lst) / len(lst) if lst else 0.0

    print("\n" + "=" * 60)
    print(f"OVO-Bench Streamo Scoring  ({jsonl_path})")
    print("=" * 60)

    cat_avgs = {}
    for cat_name, task_list in [
        ("Backward Tracing", BACKWARD_TASKS),
        ("Real-time Visual Perception", REALTIME_TASKS),
        ("Forward Active Responding", FORWARD_TASKS),
    ]:
        per = []
        print(f"\n--- {cat_name} ---")
        for t in task_list:
            if t in per_task:
                acc = _avg(per_task[t])
                per.append(acc)
                print(f"  {t}: {acc:.2f}%  ({sum(per_task[t])}/{len(per_task[t])})")
        cat_avg = sum(per) / len(per) if per else 0.0
        cat_avgs[cat_name] = cat_avg
        print(f"  {cat_name} Avg: {cat_avg:.2f}%")

    if cat_avgs:
        total = sum(cat_avgs.values()) / len(cat_avgs)
    else:
        total = 0.0
    print(f"\n  >>> Total Avg: {total:.2f}% <<<")
    print("=" * 60)

    summary = {
        "per_task": {t: _avg(v) for t, v in per_task.items()},
        "categories": cat_avgs,
        "total_avg": total,
        "n_samples": len(records),
    }
    summary_path = jsonl_path.rsplit(".", 1)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to {summary_path}")
    return summary


def batch_infer(engine: 'InferEngine', infer_requests: List['InferRequest'], batch_size: int = 16):
    if not infer_requests:
        return []
        
    request_config = RequestConfig(max_tokens=4096, temperature=0.1)
    
    all_responses = []
    for i in range(0, len(infer_requests), batch_size):
        batch_requests = infer_requests[i:i+batch_size]
        metric = InferStats()
        resp_list = engine.infer(batch_requests, request_config, metrics=[metric])
        
        batch_responses = [resp.choices[0].message.content for resp in resp_list]
        all_responses.extend(batch_responses)
        
        print(f'Batch {i//batch_size + 1}: processed {len(batch_requests)} requests')
        print(f'Metric: {metric.compute()}')
    
    return all_responses


def process_videos_batch(
    engine: "InferEngine",
    samples: List[dict],
    system: str,
    max_concurrent: int = 32,
    batch_size: int = 16,
    output_file: str = None,
    *,
    task_filter: Union[None, Set[str], List[str], str] = None,
):
    processed_ids = load_processed_ids_from_jsonl(output_file) if output_file else set()
    if processed_ids:
        orig = len(samples)
        samples = [s for s in samples if s.get("id") not in processed_ids]
        print(f"Startup filter: {orig - len(samples)} samples skipped (already processed), {len(samples)} remaining")

    task_set = _normalize_task_filter(task_filter)
    if task_set:
        orig = len(samples)
        samples = [s for s in samples if s.get("task") in task_set]
        print(f"Task filter: keep {len(samples)}/{orig} samples belonging to tasks {sorted(task_set)}")
    else:
        print("No task filter applied – all tasks will be processed.")

    if not samples:
        print("No samples to process after filtering. Exit.")
        return

    processors = {s["id"]: VideoProcessor(s, system) for s in samples}
    written_ids = set()
    eval_file = open(output_file, "a") if output_file else None

    def _advance_to_infer_or_complete(p: VideoProcessor, last_answer: str):
        skipped = 0
        la = last_answer
        while True:
            if p.is_completed():
                return None, skipped, la
            should_process, req = p.get_next_request(la)
            if should_process:
                return req, skipped, la
            p.update("</Silence>")
            skipped += 1
            la = "</Silence>"

    try:
        while True:
            active = {k: v for k, v in processors.items() if not v.is_completed()}
            if not active:
                print("All samples processed!")
                break
            print(f"\n--- Processing round ---\nActive: {len(active)}")

            batch_req, batch_keys = [], []
            for k, p in list(active.items())[:max_concurrent]:
                last_answer = p.output.get(f"Round {p.current_round-1}") if p.current_round else None
                req, skipped_cnt, _ = _advance_to_infer_or_complete(p, last_answer)

                if skipped_cnt:
                    print(
                        f"Sample-{k}: fast-forwarded {skipped_cnt} round(s) before question_time; "
                        f"now at round {p.current_round}/{p.total_rounds}"
                    )

                if req is not None:
                    batch_req.append(req)
                    batch_keys.append(k)

            if batch_req:
                responses = batch_infer(engine, batch_req, batch_size)
                for k, resp in zip(batch_keys, responses):
                    p = processors[k]
                    p.update(resp)
                    print(
                        f"Sample-{k}: id {p.sample['id']} round {p.current_round}/{p.total_rounds}, "
                        f"resp: {resp[:50]}..."
                    )

            for k, p in processors.items():
                if "gt" not in p.sample:
                    p.sample["gt"] = p.sample["test_info"]
                if p.is_completed() and eval_file and k not in written_ids:
                    final_response, trigger_round = _extract_final_response(p.output)
                    output_data = {
                        "id": p.sample["id"],
                        "video": p.sample["video"],
                        "gt": p.sample["gt"],
                        "output": p.output,
                        "task": p.sample.get("task"),
                        "final_response": final_response,
                        "trigger_round": trigger_round,
                        # GT-related context fields (for downstream analysis)
                        "question": p.sample.get("question"),
                        "options": p.sample.get("options"),
                        "test_info": p.sample.get("test_info"),
                        "realtime": p.sample.get("realtime"),
                        "question_time": p.question_time,
                        "prompt_question": p.question,
                        "total_rounds": p.total_rounds,
                    }
                    if p.sample.get("task") == "SSR":
                        output_data["step_index"] = p.sample.get("step_index", -1)
                        output_data["original_id"] = p.sample.get("original_id", "")
                        output_data["step_question"] = p.sample.get("all_steps", "")
                        output_data["step_start_time"] = p.sample.get("step_start_time", 0)
                    elif p.sample.get("task") == "CRR":
                        output_data["ask_time"] = p.sample.get("ask_time", 0)
                    elif p.sample.get("task") == "REC":
                        output_data["start_times"] = p.sample.get("start_times", [])
                        output_data["activity"] = p.sample.get("activity")
                    eval_file.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                    eval_file.flush()
                    written_ids.add(k)

            completed = sum(1 for p in processors.values() if p.is_completed())
            print(f"Progress: {completed}/{len(processors)} completed")
    finally:
        if eval_file:
            eval_file.close()



def get_data_stream_window(frame_paths, round_num, system=None, question=None, data=None, answer=None, max_rounds=120, question_time=0):
    if data is None:
        data = {}
        if round_num != 0:
            raise ValueError("round_num must be 0 when data is None")
        messages = [
                {'role': 'system', 'content': system},
            ]
        
        if round_num == question_time:
            messages.append({'role': 'user', 'content': f'{question}\n<{round_num}s-{int(round_num)+1}s>\n<image>'})
        else:
            messages.append({'role': 'user', 'content': f"<{round_num}s-{int(round_num)+1}s>\n<image>"})
        
        round_images = [
            frame_paths[round_num],
        ]
        data['images'] = round_images
        data['messages'] = messages
        return data
    else:
        messages = data['messages']
        messages.append({'role': 'assistant', 'content': answer})
        if round_num == question_time:
            messages.append({'role': 'user', 'content': f'{question}\n<{round_num}s-{int(round_num)+1}s>\n<image>'})
        else:
            messages.append({'role': 'user', 'content': f"<{round_num}s-{int(round_num)+1}s>\n<image>"})
        
        round_images = [
            frame_paths[round_num],
        ]
        data['images'].extend(round_images)
        
        if round_num > max_rounds:

            rounds_to_remove = 1
            
            new_messages = messages[:1]
            messages_to_skip = rounds_to_remove * 2
            
            new_messages.extend(messages[1 + messages_to_skip:])
            
            new_messages[1] = {'role': 'user', 'content': f'{question}\n<{round_num-max_rounds}s-{int(round_num)-max_rounds+1}s>\n<image>'}

            images_to_remove = rounds_to_remove
            new_images = data['images'][images_to_remove:]
            
            data['messages'] = new_messages
            data['images'] = new_images
        
        return data


def is_video_directory(directory_path):
    if not os.path.isdir(directory_path):
        return False
    
    frame_patterns = [
        r'frame_\d{6}\.(png|jpg|jpeg)',
        r'frame_\d{5}\.(png|jpg|jpeg)',
        r'frame_\d{4}\.(png|jpg|jpeg)',
        r'\d{6}\.(png|jpg|jpeg)',
        r'\d{5}\.(png|jpg|jpeg)',
    ]
    
    try:
        files = os.listdir(directory_path)
        frame_files = []
        
        for file in files:
            for pattern in frame_patterns:
                if re.match(pattern, file, re.IGNORECASE):
                    frame_files.append(file)
                    break
        
        return len(frame_files) >= 2
    
    except (PermissionError, OSError):
        return False


def find_video_directories(root_path, max_depth=10):
    video_dirs = []
    
    def _recursive_search(current_path, current_depth):
        if current_depth > max_depth:
            return
        
        try:
            if not os.path.isdir(current_path):
                return
            
            if is_video_directory(current_path):
                video_dirs.append(current_path)
                print(f"Found video directory: {current_path}")
                return
            
            for item in os.listdir(current_path):
                item_path = os.path.join(current_path, item)
                if os.path.isdir(item_path):
                    _recursive_search(item_path, current_depth + 1)
                    
        except (PermissionError, OSError) as e:
            print(f"Warning: Cannot access {current_path}: {e}")
    
    print(f"Searching for video directories in: {root_path}")
    _recursive_search(root_path, 0)
    print(f"Found {len(video_dirs)} video directories total")
    
    return video_dirs


if __name__ == "__main__":
    from swift import (
        InferEngine,
        InferRequest,
        InferStats,
    )
    from swift.infer_engine import (
        VllmEngine,
        RequestConfig,
    )

    parser = argparse.ArgumentParser(description="OVOBench video stream inference batch processing")
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Path to OVO-Bench dataset JSON",
    )
    parser.add_argument(
        "--video_prefix",
        type=str,
        default="",
        help="Prefix to prepend to video paths in dataset",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=["EPM", "HLD", "ASI", "STU", "OJR", "ATR", "FPD", "OCR", "SSR", "ACR", "CRR", "REC"],
        help="Comma-separated task list to process, e.g.: EPM,SSR. If not specified, all tasks will be processed.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./test.jsonl",
        help="Output file path (jsonl)",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=64,
        help="Maximum number of samples to process concurrently per round",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for model inference",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.7,
        help="GPU memory utilization for vLLM",
    )
    parser.add_argument(
        "--rec_prompt_mode",
        type=str,
        default="letter",
        choices=["letter", "number"],
        help="REC task prompt: 'letter' (A-K MCQ -> 0..10) or 'number' (free-form integer 0..10).",
    )
    args = parser.parse_args()

    REC_PROMPT_MODE = args.rec_prompt_mode  # noqa: F841 (kept for clarity)
    globals()["REC_PROMPT_MODE"] = args.rec_prompt_mode

    infer_backend = "vllm"
    model = args.model
    engine = VllmEngine(
        model,
        max_model_len=131072,
        limit_mm_per_prompt={"image": 500},
        tensor_parallel_size=args.tp,
        enable_prefix_caching=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    dataset_json = args.dataset
    samples = load_dataset_json(dataset_json)

    # Set video prefix for path remapping
    VIDEO_PREFIX = args.video_prefix

    system = SYSTEM

    process_videos_batch(
        engine=engine,
        samples=samples,
        system=system,
        max_concurrent=args.max_concurrent,
        batch_size=args.batch_size,
        output_file=args.output,
        task_filter=args.tasks,
    )

    # Unified scoring after inference completes.
    score_results(args.output)
