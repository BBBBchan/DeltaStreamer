#!/usr/bin/env python
"""
Shared-queue worker for OVO-Bench agentic evaluation.

设计目标
========
- 4 个 worker(每卡 1 个 TP=1 vLLM 实例)动态共享同一份 sample 列表;
- 用 POSIX O_EXCL 文件原子创建做认领(无外部依赖、NFS-safe);
- 每 worker 每次认领 K 个样本, 跑完再认领下一批 → 真动态负载均衡,
  不存在"长样本扎堆某 worker 拖尾, 其它卡闲置"的问题;
- 复用 agent.process_videos_agent 现有逻辑(in-flight 池仍生效),
  本脚本只是把 samples 列表替换成"动态认领的子集";
- 每 worker 写独立的 *.worker<N>.jsonl, 由 shell 脚本最后合并。

CLI
===
继承 agent.py 的所有参数, 额外新增:
  --worker_id N        本 worker 编号(0-based, 仅用于日志/claim 文件标记)
  --num_workers N      worker 总数(仅用于日志)
  --claims_dir PATH    共享认领目录(每个 sample 一个 .claim 文件)
  --claim_batch_size K 每批认领样本数(默认 16)
  --skip_score         跳过末尾 score_results(由合并脚本统一打分)
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from typing import List, Set

# 让脚本在仓库根/scripts 下都能 import 到 agent / video_processor
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import agent  # noqa: E402
import video_processor as ovo  # noqa: E402
from agent import (  # noqa: E402
    Budget,
    build_default_registry,
    build_extended_registry,
    process_videos_agent,
    render_agent_system,
)


# ----------------------------------------------------------------------------
# 原子认领
# ----------------------------------------------------------------------------
def atomic_claim_batch(
    claims_dir: str,
    candidates: List[str],
    batch_size: int,
    worker_id: int,
) -> List[str]:
    """对 ``candidates`` 顺序尝试 O_EXCL 创建 .claim 文件,
    成功的 id 计为本 worker 认领。最多认领 ``batch_size`` 个。
    """
    os.makedirs(claims_dir, exist_ok=True)
    claimed: List[str] = []
    for sid in candidates:
        if len(claimed) >= batch_size:
            break
        path = os.path.join(claims_dir, f"{sid}.claim")
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            try:
                os.write(
                    fd,
                    (
                        f"worker={worker_id}\n"
                        f"pid={os.getpid()}\n"
                        f"ts={time.time()}\n"
                    ).encode(),
                )
            finally:
                os.close(fd)
            claimed.append(sid)
        except FileExistsError:
            continue
    return claimed


def load_global_processed_ids(output_dir: str, run_tag: str) -> Set[str]:
    """读所有 worker 已写出的 jsonl, 收集已完成的 sample id 集合。"""
    import glob as _g
    done: Set[str] = set()
    output_prefix = os.environ.get("STREAMO_OUTPUT_PREFIX", "ovobench")
    for fp in _g.glob(os.path.join(
        output_dir, f"{output_prefix}_{run_tag}.worker*.jsonl"
    )):
        try:
            with open(fp) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                        sid = r.get("id")
                        if sid is not None:
                            done.add(str(sid))
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            continue
    return done


def pending_claims(
    claims_dir: str,
    all_ids: List[str],
    done: Set[str],
) -> List[str]:
    return [
        sid
        for sid in all_ids
        if sid not in done
        and os.path.exists(os.path.join(claims_dir, f"{sid}.claim"))
    ]


def claim_owner_alive(claim_path: str):
    """Return True/False for PID-aware claims, None for legacy claims."""
    try:
        with open(claim_path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return None
    pid = None
    for line in text.splitlines():
        if line.startswith("pid="):
            try:
                pid = int(line.split("=", 1)[1])
            except ValueError:
                return None
            break
    if pid is None:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ----------------------------------------------------------------------------
# CLI(扩展 agent 的 argparser)
# ----------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = agent._build_argparser()
    p.description = "OVO-Bench shared-queue agent worker."
    g = p.add_argument_group("shared-queue worker options")
    g.add_argument("--worker_id", type=int, required=True,
                   help="0-based worker index, used in log + claim record")
    g.add_argument("--num_workers", type=int, default=4,
                   help="total worker count (informational only)")
    g.add_argument("--claims_dir", required=True,
                   help="shared directory holding <id>.claim atomic stubs")
    g.add_argument("--claim_batch_size", type=int, default=16,
                   help="number of samples each worker claims per round")
    g.add_argument("--run_tag", required=True,
                   help="run tag used for partial output filename")
    g.add_argument("--output_dir", required=True,
                   help="directory to write <prefix>_<tag>.worker<id>.jsonl")
    g.add_argument("--output_prefix", default="ovobench",
                   help="result shard filename prefix")
    g.add_argument("--skip_score", action="store_true",
                   help="skip ovo.score_results at end (merge step does it)")
    return p


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    args = _build_argparser().parse_args()
    os.environ["STREAMO_OUTPUT_PREFIX"] = args.output_prefix

    # 1) 引擎初始化(同 agent.main, 保持 TP/gpu_mem 等行为一致)
    from swift import VllmEngine
    # Ascend NPU 上 torch.compile / aclgraph 捕获不稳定（qwen3_5 在 aot_compile
    # 阶段报 dynamo TypeError），默认走 enforce_eager；可用 STREAMO_ENFORCE_EAGER=0 关闭。
    _enforce_eager = os.environ.get("STREAMO_ENFORCE_EAGER", "1") not in ("0", "false", "False")
    # Qwen3.5 混合架构的 GDN(gated delta net) prefill 内核，vLLM 默认(auto)在 SM90(Hopper)
    # 上选 flashinfer JIT 内核；但 flashinfer 0.6.6 打包的 SM90 模板有编译期 bug
    # (flat::Unused 缺 consumer_wait/index 等成员 → Ninja/nvcc 编译失败)，warmup 在
    # try/except 里被吞成 WARNING，worker 反复编译空耗 GPU 起不来。改走 triton 后端
    # (forward_native，无需 nvcc JIT)绕开。可用 STREAMO_GDN_PREFILL_BACKEND 覆盖。
    _gdn_backend = os.environ.get("STREAMO_GDN_PREFILL_BACKEND", "triton").strip().lower()
    _engine_kwargs = {}
    if _gdn_backend in ("triton", "flashinfer"):
        _engine_kwargs["gdn_prefill_backend"] = _gdn_backend
    print(f"[worker{args.worker_id}] gdn_prefill_backend={_gdn_backend or 'auto(default)'}")
    engine = VllmEngine(
        args.model,
        max_model_len=65536,
        limit_mm_per_prompt={"image": 600, "video": 1},
        tensor_parallel_size=args.tp,
        enable_prefix_caching=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=_enforce_eager,
        engine_kwargs=_engine_kwargs,
    )
    if args.no_think:
        try:
            engine.template.enable_thinking = False
            print(f"[worker{args.worker_id}] no_think enabled")
        except Exception as e:
            print(f"[worker{args.worker_id}] no_think failed: {e}")

    # 2) 加载完整 samples + task_filter
    ovo.VIDEO_PREFIX = args.video_prefix
    samples = ovo.load_dataset_json(args.dataset)
    task_set = ovo._normalize_task_filter(args.tasks)
    if task_set:
        samples = [s for s in samples if s.get("task") in task_set]
    sample_by_id = {str(s["id"]): s for s in samples}
    all_ids: List[str] = [str(s["id"]) for s in samples]
    print(f"[worker{args.worker_id}] loaded {len(samples)} samples "
          f"(tasks={sorted(task_set) if task_set else 'all'})")

    # 3) registry / system / budget 工厂(与 agent.main 一致)
    use_extended = (
        args.memory_mode == "brief"
        or args.enable_memory_tools
        or args.enable_track_tools
        or args.enable_audio_tools
        or args.enable_efficiency_tools
    )
    if use_extended:
        registry = build_extended_registry(
            enable_memory=args.enable_memory_tools or args.memory_mode == "brief",
            enable_track=args.enable_track_tools,
            enable_audio=args.enable_audio_tools,
            enable_efficiency=args.enable_efficiency_tools or args.memory_mode == "brief",
        )
    else:
        registry = build_default_registry()
    base_system = ovo.SYSTEM if hasattr(ovo, "SYSTEM") else ""
    system = (
        base_system if args.no_tools
        else render_agent_system(registry, base_system, memory_mode=args.memory_mode,
                                  delta_input_mode=getattr(args, 'delta_input_mode', '2frame'))
    )

    def _budget_factory():
        return Budget(
            max_tool_calls=args.budget_max_tool_calls,
            max_frames_seen=args.budget_max_frames_seen,
            max_tokens=args.budget_max_tokens,
        )

    # 4) 输出文件:每 worker 自己一份分片
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f"{args.output_prefix}_{args.run_tag}.worker{args.worker_id}.jsonl",
    )
    print(f"[worker{args.worker_id}] output → {out_path}")

    # 5) 共享队列主循环
    rounds_done = 0
    samples_done = 0
    claim_wait_started = None
    claim_wait_seconds = float(os.environ.get(
        "STREAMO_CLAIM_WAIT_SECONDS", "21600"
    ))
    while True:
        # 全局已完成(任一 worker 写出过) → 不再认领
        global_done = load_global_processed_ids(args.output_dir, args.run_tag)
        # 候选 = 全部样本 - 全局已完成 - 已被任何 worker 认领过的
        candidates = [
            sid for sid in all_ids
            if sid not in global_done
            and not os.path.exists(os.path.join(args.claims_dir, f"{sid}.claim"))
        ]
        if not candidates:
            pending = pending_claims(args.claims_dir, all_ids, global_done)
            if pending:
                now = time.time()
                claim_wait_started = claim_wait_started or now
                owners = [
                    claim_owner_alive(
                        os.path.join(args.claims_dir, f"{sid}.claim")
                    )
                    for sid in pending
                ]
                elapsed = now - claim_wait_started
                if elapsed < claim_wait_seconds and any(
                    owner is not False for owner in owners
                ):
                    print(
                        f"[worker{args.worker_id}] waiting for "
                        f"{len(pending)} claimed samples "
                        f"(alive={owners.count(True)}, "
                        f"legacy={owners.count(None)}, "
                        f"elapsed={elapsed:.0f}s)"
                    )
                    time.sleep(5)
                    continue
                print(
                    f"[worker{args.worker_id}] pending claims did not "
                    f"complete before exit: {pending[:12]}"
                )
            else:
                print(
                    f"[worker{args.worker_id}] no candidates or pending "
                    f"claims left; exit. rounds={rounds_done} "
                    f"samples={samples_done}"
                )
            break
        claim_wait_started = None

        claimed_ids = atomic_claim_batch(
            args.claims_dir, candidates, args.claim_batch_size, args.worker_id,
        )
        if not claimed_ids:
            # 这一轮全被别的 worker 抢走了, 等一下再扫
            time.sleep(2)
            continue

        sub_samples = [sample_by_id[i] for i in claimed_ids]
        rounds_done += 1
        samples_done += len(sub_samples)
        print(f"[worker{args.worker_id}] round {rounds_done}: "
              f"claimed {len(sub_samples)} samples → "
              f"{[s.get('task')+'/'+str(s.get('id')) for s in sub_samples[:6]]}"
              f"{' ...' if len(sub_samples)>6 else ''}")

        # 调用现有 process_videos_agent 跑这一批
        # 注意:它内部会 a-mode open(out_path), 直接 append; 多 worker
        # 写不同 jsonl 文件, 所以不会冲突。
        process_videos_agent(
            engine=engine,
            samples=sub_samples,
            system=system,
            registry=registry,
            max_concurrent=args.max_concurrent,
            batch_size=args.batch_size,
            output_file=out_path,
            task_filter=None,  # 已在外层过滤过, 这里禁用避免误删
            max_tool_calls_per_round=args.max_tool_calls_per_round,
            max_tool_calls_per_sample=args.max_tool_calls_per_sample,
            max_tool_calls_per_reply=args.max_tool_calls_per_reply,
            memory_mode=args.memory_mode,
            memory_token_budget=args.memory_token_budget,
            budget_factory=_budget_factory,
            trajectory_dir=args.trajectory_dir,
            auto_anchor_every=args.auto_anchor_every,
            auto_asr_every=args.auto_asr_every,
            auto_asr_window=args.auto_asr_window,
            auto_track_query_every=args.auto_track_query_every,
            memory_render_mode=args.memory_render_mode,
            memory_head_max=args.memory_head_max,
            memory_tail_max=args.memory_tail_max,
            memory_head_seconds=args.memory_head_seconds,
            memory_tail_seconds=args.memory_tail_seconds,
            memory_recall_topk=args.memory_recall_topk,
            cold_start_rounds=args.cold_start_rounds,
            degrade_threshold=args.degrade_threshold,
            degrade_events_min=args.degrade_events_min,
            rolling_window=args.rolling_window,
            delta_input_mode=getattr(args, 'delta_input_mode', '2frame'),
            enable_verification=getattr(args, 'enable_verification', False),
        )

    if not args.skip_score:
        ovo.score_results(out_path)


if __name__ == "__main__":
    main()
