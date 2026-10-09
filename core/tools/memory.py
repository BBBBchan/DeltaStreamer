"""Memory store + memory_* tools.

Phase-A scaffolding (no external services). Provides:

* :class:`MemoryStore`: per-sample structured memory (events / entities /
  tracks / audio_segments) with a ``snapshot()`` method that produces a
  bounded-length textual brief for prompt injection.
* :class:`RuleSummarizer`: a zero-dep summarizer that turns a model
  response + frame index into a 1–2 sentence event for the store.
* Tools: ``memory_write``, ``memory_read``, ``memory_list``, ``memory_forget``.

This module is *additive*: ``video_processor.py`` / ``video_processor_api.py`` are
not affected. The agent main loop opts in by passing a ``MemoryStore`` via
:class:`ToolContext` and gating prompt construction on ``ctx.memory``.
"""
from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from .base import Tool, ToolContext, ToolResult


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Event:
    eid: int
    t: int
    kind: str            # observation | qa | tool | audio | note
    text: str
    ref_frame: Optional[int] = None
    salience: float = 1.0
    ts: float = field(default_factory=time.time)
    # ---- tree-structure / change-tracking fields -----------------------
    delta: Optional[str] = None              # change snippet from <Observation>
    is_change: bool = False                  # frame is a "change" event
    parent_episode_id: Optional[int] = None  # back-pointer to episodic layer
    entities: List[str] = field(default_factory=list)  # extracted entity slugs

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.eid, "t": self.t, "kind": self.kind,
            "text": self.text, "ref_frame": self.ref_frame,
            "salience": round(self.salience, 3),
        }
        if self.delta is not None:
            d["delta"] = self.delta
        if self.is_change:
            d["is_change"] = True
        if self.parent_episode_id is not None:
            d["parent_episode_id"] = self.parent_episode_id
        return d


def _split_fields(text: str) -> Dict[str, str]:
    """Best-effort parse of a labelled OBSERVE-ONLY body into a dict.

    Handles both the multi-line ``label: value\\n`` format and the
    single-paragraph ``label: value ; label: value`` form. Unknown
    labels are dropped. Values are stripped.

    Also handles bullet-prefixed formats like ``• description: ...``
    which the model commonly emits in OBSERVE-ONLY mode.
    """
    if not text:
        return {}
    # Strip bullet markers ("•", "-", "*") that prefix field labels.
    # The MULTILINE flag handles per-line bullets ("\\n• field:"); we
    # additionally replace " ; •" with " ;" so semicolon-joined
    # bullet fields (e.g. copied from a multi-line source) are also
    # normalised.
    text = text.replace("\r", "")
    text = re.sub(r'^[•\-*]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*;\s*[•\-*]\s*', ' ; ', text)
    # Normalise both forms onto ``;`` separators so we can split once.
    norm = text.replace("\n", " ; ")
    out: Dict[str, str] = {}
    cur_key: Optional[str] = None
    buf: List[str] = []
    for chunk in norm.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.match(
            r"^(description|actors?|objects?|actions?|ocr|"
            r"text_on_screen|scene|delta|actor|action)\s*[:=]\s*(.*)$",
            chunk, flags=re.IGNORECASE,
        )
        if m:
            if cur_key and buf:
                out.setdefault(cur_key, " ; ".join(buf).strip())
            cur_key = m.group(1).lower().rstrip("s")
            # canonical forms
            if cur_key == "text_on_screen":
                cur_key = "ocr"
            elif cur_key == "actor":
                cur_key = "actor"
            elif cur_key == "action":
                cur_key = "action"
            buf = [m.group(2).strip()]
        else:
            if cur_key:
                buf.append(chunk)
    if cur_key and buf:
        out.setdefault(cur_key, " ; ".join(buf).strip())
    return out


def _summarize_run(run: List["Event"]) -> str:
    """Build a real time-window summary of a run of events.

    The episodic layer is supposed to "tell future-you what happened
    across this whole [t0,t1] period" without re-emitting every
    single frame. This is a rule-based summarizer (no LLM) that:

    * synthesises ONE ``description`` for the whole window: if all
      per-frame descriptions are identical we say "全程: ..."; if
      they evolve we keep distinct ones in chronological order
      (head → mid → tail) so future-you sees the period's arc;
    * unions the scene / actors / objects / actions / ocr fields
      across the window so structured state is preserved;
    * extracts the non-static ``delta`` snippets, dedups them, and
      surfaces them as ``changes`` so the period's actual motion is
      captured without per-second clutter.

    No per-frame timestamps appear in the output — that level of
    detail belongs to the working layer. ``write_episode`` caps the
    final string at 6000 chars.
    """
    if not run:
        return ""

    # Pass 1: parse every event's body into labelled fields.
    parsed: List[Tuple[int, Dict[str, str], str]] = []
    for ev in run:
        text = (ev.text or "").strip()
        fields = _split_fields(text)
        parsed.append((ev.t, fields, text))

    # Pass 2: union-aggregate repeating fields.
    def _collect(label: str) -> List[str]:
        seen: List[str] = []
        for _, fields, _txt in parsed:
            v = (fields.get(label) or "").strip()
            if not v:
                continue
            v_norm = v.lower()
            if any(v_norm == s.lower() for s in seen):
                continue
            seen.append(v)
        return seen

    actors_u = _collect("actor")
    objects_u = _collect("object")
    actions_u = _collect("action")
    scene_u = _collect("scene")
    ocr_u = [v for v in _collect("ocr") if v.lower() not in ("(none)", "none")]

    # Synthesise a single-window description without per-frame stamps.
    # If every frame says the same thing → 全程: ...
    # Otherwise keep ≤3 distinct descriptions in chronological order.
    descs_in_order: List[str] = []
    seen_desc_norm: set = set()
    for _, fields, _txt in parsed:
        d = (fields.get("description") or "").strip()
        if not d:
            continue
        key = d.lower()
        if key in seen_desc_norm:
            continue
        seen_desc_norm.add(key)
        descs_in_order.append(d)
    if len(descs_in_order) == 0:
        narrative = ""
    elif len(descs_in_order) == 1:
        narrative = f"throughout: {descs_in_order[0]}"
    else:
        # Keep head, optional mid, tail to convey the arc.
        if len(descs_in_order) == 2:
            picks = descs_in_order
        else:
            picks = [
                descs_in_order[0],
                descs_in_order[len(descs_in_order) // 2],
                descs_in_order[-1],
            ]
        narrative = " → ".join(p[:240] for p in picks)

    # Pass 3: collect non-static delta snippets as the "changes" log.
    static_re = re.compile(
        r"^\s*(static|no change|visually identical)\b", re.IGNORECASE,
    )
    changes_seen: List[str] = []
    n_static = 0
    for _, fields, _txt in parsed:
        dl = (fields.get("delta") or "").strip()
        if not dl:
            continue
        if static_re.match(dl):
            n_static += 1
            continue
        norm = dl.lower()
        if any(norm == c.lower() for c in changes_seen):
            continue
        changes_seen.append(dl)

    t0 = run[0].t
    t1 = run[-1].t
    span = max(1, t1 - t0 + 1)

    parts: List[str] = []
    parts.append(f"summary of [t={t0}-{t1}s] ({span}s, {len(run)} obs):")
    if narrative:
        parts.append(f"  description: {narrative[:1200]}")
    if scene_u:
        parts.append("  scene: " + " → ".join(s[:140] for s in scene_u[:8]))
    if actors_u:
        parts.append("  actors (union): "
                     + " | ".join(a[:120] for a in actors_u[:12]))
    if objects_u:
        parts.append("  objects (union): "
                     + " | ".join(o[:160] for o in objects_u[:16]))
    if actions_u:
        parts.append("  actions (union): "
                     + " | ".join(a[:120] for a in actions_u[:12]))
    if ocr_u:
        parts.append("  ocr (union): " + " | ".join(o[:80] for o in ocr_u[:12]))
    if changes_seen:
        change_lines = [c[:200] for c in changes_seen[:20]]
        parts.append("  changes: " + " | ".join(change_lines))
    elif n_static > 0:
        parts.append(f"  changes: (none reported across {n_static} obs — verified static)")

    # Fallback if nothing parsed (legacy events with no labels).
    if len(parts) <= 1:
        snippets: List[str] = []
        n = len(parsed)
        if n <= 8:
            picks = parsed
        else:
            mid_n = 6
            mid = [round(i * (n - 1) / (mid_n + 1)) for i in range(1, mid_n + 1)]
            idxs = sorted(set([0] + mid + [n - 1]))
            picks = [parsed[i] for i in idxs]
        for t, _f, txt in picks:
            snippet = (txt or "").strip()[:360]
            if snippet:
                snippets.append(f"[t={t}s] {snippet}")
        return " | ".join(snippets)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Episode-run signature & similarity (used to merge adjacent runs whose
# observations describe nearly identical content). The episodic layer can
# otherwise emit dozens of micro-episodes that all read "throughout: A
# high-angle view of the kitchen counter ..." simply because the model
# kept producing the same labels frame-after-frame. The merge happens
# AFTER ``auto_aggregate_episodes`` has cut runs by gap/span and BEFORE
# they are written as ``episodes``.
# ---------------------------------------------------------------------------
_SIG_FIELDS = ("description", "scene", "actor", "object", "action", "ocr")
_SIG_TOK_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")
_SIG_STOPWORDS = frozenset({
    "the", "a", "an", "of", "on", "in", "at", "to", "and", "or", "is",
    "are", "with", "for", "by", "as", "from", "this", "that", "it",
    "be", "been", "was", "were", "has", "have", "had", "but", "not",
    "no", "yes", "none", "n", "s", "very", "above", "below", "into",
    "onto", "over", "under", "across",
})


def _run_signature(run: List["Event"]) -> Dict[str, set]:
    """Produce a per-field token set fingerprint for an event run.

    Tokens are lower-cased, stop-words removed, single-char tokens
    dropped. Empty fields yield empty sets.
    """
    sig: Dict[str, set] = {f: set() for f in _SIG_FIELDS}
    for ev in run:
        text = (ev.text or "")
        # Strip the [template-copy] marker the summarizer may have
        # prepended so two runs that differ only by that marker are
        # still recognised as similar.
        if text.startswith("[template-copy]"):
            text = text[len("[template-copy]"):].lstrip()
        fields = _split_fields(text)
        for f in _SIG_FIELDS:
            v = (fields.get(f) or "").strip().lower()
            if not v or v in ("(none)", "none"):
                continue
            for tok in _SIG_TOK_RE.findall(v):
                if len(tok) <= 1 or tok in _SIG_STOPWORDS:
                    continue
                sig[f].add(tok)
    return sig


def _run_similarity(sig_a: Dict[str, set], sig_b: Dict[str, set]) -> float:
    """Mean Jaccard across the signature fields.

    Fields where both sides are empty are skipped (don't punish).
    Fields where exactly one side is empty contribute 0.
    """
    sims: List[float] = []
    for f in _SIG_FIELDS:
        a = sig_a.get(f) or set()
        b = sig_b.get(f) or set()
        if not a and not b:
            continue
        if not a or not b:
            sims.append(0.0)
            continue
        inter = len(a & b)
        union = len(a | b)
        sims.append(inter / union if union else 0.0)
    if not sims:
        return 0.0
    return sum(sims) / len(sims)


def _merge_runs_by_similarity(
    runs: List[List["Event"]],
    threshold: float = 0.85,
    max_gap: int = 30,
) -> List[List["Event"]]:
    """Iteratively merge adjacent runs whose signatures are very similar.

    Two adjacent runs (run_a, run_b) are merged when:
      * the gap between ``run_a[-1].t`` and ``run_b[0].t`` is
        ``<= max_gap`` (so we don't bridge truly disjoint segments),
      * their content signatures' mean Jaccard ``>= threshold``.

    The pass repeats until no more merges happen — this lets a chain of
    similar runs collapse into a single episode covering their full span.
    """
    if not runs:
        return runs
    while True:
        merged_any = False
        out: List[List["Event"]] = []
        i = 0
        while i < len(runs):
            cur = runs[i]
            if i + 1 < len(runs):
                nxt = runs[i + 1]
                gap = nxt[0].t - cur[-1].t
                if 0 <= gap <= int(max_gap):
                    sim = _run_similarity(
                        _run_signature(cur), _run_signature(nxt))
                    if sim >= float(threshold):
                        out.append(cur + nxt)
                        i += 2
                        merged_any = True
                        continue
            out.append(cur)
            i += 1
        runs = out
        if not merged_any:
            break
    return runs


@dataclass
class MemoryStore:
    sample_id: str
    events: List[Event] = field(default_factory=list)
    entities: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tracks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    audio_segments: List[Dict[str, Any]] = field(default_factory=list)
    # Long-term layer: pinned facts, never evicted, always rendered first.
    # Each item: {"id": int, "text": str, "kind": str, "t": Optional[int]}
    pinned_facts: List[Dict[str, Any]] = field(default_factory=list)
    # Episodic layer: aggregated events spanning a [t0, t1] window with a
    # 1-2 sentence summary. Built either explicitly via write_episode() or
    # automatically by auto_aggregate_episodes() at snapshot time.
    # Each item: {"id": int, "t0": int, "t1": int, "kind": str,
    #             "summary": str, "ev_ids": List[int], "salience": float}
    episodes: List[Dict[str, Any]] = field(default_factory=list)
    _next_eid: int = 1
    _next_pin_id: int = 1
    _next_epid: int = 1
    # Optional LLM-based summarizer. When set, ``auto_aggregate_episodes``
    # and ``_seal_anchor_run`` call it first to produce a 1-3 sentence
    # window summary; on failure they silently fall back to the rule-based
    # ``_summarize_run``. The callable receives the run (List[Event]) and
    # returns either a non-empty summary string or None.
    summarizer_fn: Optional[Any] = None
    # ── Hybrid retrieval cache ───────────────────────────────────────
    # Per-store embedding caches populated by post_graph_hook; used by
    # ``_hybrid_score()`` to blend keyword overlap with embedding cosine.
    _query_embed_fn: Optional[Any] = field(default=None, repr=False)
    _event_embeddings: Dict[int, Any] = field(default_factory=dict, repr=False)

    def _make_summary(self, run: "List[Event]") -> str:
        """Build an episode summary, preferring ``summarizer_fn`` when
        provided. Always returns a non-empty string (worst case empty).
        """
        if self.summarizer_fn is not None:
            try:
                out = self.summarizer_fn(run)
            except Exception:
                out = None
            if out:
                return str(out)[:6000]
        return _summarize_run(run)

    # ---- write paths --------------------------------------------------
    def write_event(
        self,
        t: int,
        kind: str,
        text: str,
        ref_frame: Optional[int] = None,
        salience: float = 1.0,
        delta: Optional[str] = None,
        is_change: bool = False,
        entities: Optional[List[str]] = None,
    ) -> Event:
        # Preserve full observation content (task / scene / actions / objects
        # / spatial / delta …). 4000 chars is large enough to cover any
        # single-frame observation we have seen in practice while still
        # bounding pathological model output.
        ev = Event(
            eid=self._next_eid, t=int(t), kind=str(kind),
            text=str(text or "")[:4000], ref_frame=ref_frame,
            salience=float(salience),
            delta=(str(delta)[:240] if delta else None),
            is_change=bool(is_change),
            entities=list(entities) if entities else [],
        )
        self._next_eid += 1
        self.events.append(ev)
        return ev

    def update_entity(self, eid: str, **fields) -> None:
        ent = self.entities.setdefault(eid, {"id": eid})
        ent.update(fields)

    def upsert_track(self, track_id: str, **fields) -> None:
        tr = self.tracks.setdefault(track_id, {"id": track_id, "traj": []})
        for k, v in fields.items():
            if k == "traj_append" and isinstance(v, list):
                tr["traj"].extend(v)
            else:
                tr[k] = v

    def add_audio(self, t0: int, t1: int, text: str,
                  speaker: Optional[str] = None) -> None:
        self.audio_segments.append({
            "t0": int(t0), "t1": int(t1),
            "text": str(text or "")[:512],
            "speaker": speaker,
        })

    def forget(self, eid: int) -> bool:
        before = len(self.events)
        self.events = [e for e in self.events if e.eid != eid]
        return len(self.events) < before

    # ---- pinned facts (long-term memory) ------------------------------
    def pin_fact(self, text: str, kind: str = "fact",
                 t: Optional[int] = None,
                 max_len: Optional[int] = None,
                 ep_id: Optional[int] = None,
                 subject: Optional[str] = None) -> Dict[str, Any]:
        # Recaps are 3–8-sentence model-authored consolidations; they
        # need much more room than a one-line fact. Default budgets:
        #   recap* / boundary → 1200 chars (model-authored, fact-covering)
        #   others            → 240 chars (legacy)
        if max_len is None:
            _k = str(kind)
            max_len = (1200 if (_k.startswith("recap") or _k == "boundary")
                       else 240)
        item = {
            "id": self._next_pin_id,
            "text": str(text or "")[:int(max_len)],
            "kind": str(kind),
            "t": (int(t) if t is not None else None),
            "ep_id": (int(ep_id) if ep_id is not None else None),
            # Subject-based supersession key (None == no key, never
            # supersedes). Normalised to a short slug by the parser.
            "subject": (str(subject)[:48] if subject else None),
            # Set when a later pin with the same (kind_family, subject)
            # appears. Superseded items are kept in storage (so
            # ``memory_search`` can still surface them) but moved to
            # the "archived" section at render time.
            "superseded_by": None,
            "superseded_at": None,
        }
        self._next_pin_id += 1
        # ---- supersession -------------------------------------------------
        # Only fire when the new pin has a non-empty subject. ``kind``
        # is collapsed to its family ("recap_object_change" → "object_change",
        # bare "fact"/"spatial"/... pass through) so the matcher tolerates
        # the recap_ prefix that ``_maybe_pin_recaps`` adds.
        if item["subject"]:
            new_family = self._kind_family(item["kind"])
            for prev in self.pinned_facts:
                if prev.get("superseded_by") is not None:
                    continue
                if not prev.get("subject"):
                    continue
                if prev["subject"] != item["subject"]:
                    continue
                if self._kind_family(prev.get("kind") or "") != new_family:
                    continue
                prev["superseded_by"] = item["id"]
                prev["superseded_at"] = item["t"]
        self.pinned_facts.append(item)
        # Mirror back-pointer on the source episode (if any).
        if ep_id is not None:
            for ep in self.episodes:
                if ep.get("id") == ep_id:
                    ep["pin_id"] = item["id"]
                    break
        return item

    @staticmethod
    def _kind_family(kind: str) -> str:
        """Strip the leading ``recap_`` prefix so e.g.
        ``recap_object_change`` and a hypothetical bare
        ``object_change`` collapse to the same supersession bucket.
        """
        k = str(kind or "")
        if k.startswith("recap_"):
            return k[len("recap_"):]
        return k

    def unpin_fact(self, pid: int) -> bool:
        before = len(self.pinned_facts)
        self.pinned_facts = [p for p in self.pinned_facts if p["id"] != pid]
        return len(self.pinned_facts) < before

    def ensure_episodic_coverage(self,
                                  current_t: int,
                                  max_gap_seconds: int = 30,
                                  tail_protect: int = 8,
                                  min_run: int = 2,
                                  max_episode_span: int = 30) -> int:
        """Time-coverage fallback for the episodic layer.

        ``auto_aggregate_episodes`` requires at least one ``is_change``
        event in a run, which causes long stretches of pure observation
        (or rounds where the model emitted ``delta=(no change)``) to
        never become episodes. The result is an episodic layer whose
        timeline ends far in the past, leaving a 30s+ blind spot
        between the last episode and ``current_t``.

        This method backfills that gap. We look at the longest
        un-covered span ending at ``current_t - tail_protect`` (the
        ``tail_protect`` window is intentionally left to the working
        layer's verbatim tail). If that span exceeds
        ``max_gap_seconds`` and contains at least ``min_run`` events,
        we emit one or more "anchor" episodes covering it — even when
        no event in the run carries ``is_change=True``. The summary
        concatenates the head and tail event texts so the layer at
        least preserves the shape of what happened in that span.

        Returns the number of episodes newly created.
        """
        if not self.events:
            return 0
        # 1) Build the set of event ids already wrapped by an episode.
        covered = set()
        last_ep_t1 = -1
        for ep in self.episodes:
            covered.update(ep.get("ev_ids") or [])
            try:
                last_ep_t1 = max(last_ep_t1, int(ep.get("t1") or -1))
            except (TypeError, ValueError):
                pass
        # 2) Determine the gap [gap_lo, gap_hi]. Anything past
        #    ``current_t - tail_protect`` is left to the working tail.
        gap_lo = max(0, last_ep_t1 + 1)
        gap_hi = max(0, int(current_t) - int(tail_protect))
        if gap_hi - gap_lo < int(max_gap_seconds):
            return 0
        # 3) Collect un-covered events inside [gap_lo, gap_hi].
        evs = sorted(
            (e for e in self.events
             if e.eid not in covered and gap_lo <= e.t <= gap_hi),
            key=lambda e: e.t,
        )
        if len(evs) < int(min_run):
            return 0
        # 4) Slice into runs not exceeding ``max_episode_span``.
        runs: List[List[Event]] = []
        cur: List[Event] = []
        for e in evs:
            if not cur:
                cur = [e]
                continue
            if (e.t - cur[0].t) > int(max_episode_span):
                if len(cur) >= int(min_run):
                    runs.append(cur)
                cur = [e]
            else:
                cur.append(e)
        if len(cur) >= int(min_run):
            runs.append(cur)
        # Same content-similarity merge as ``auto_aggregate_episodes``:
        # otherwise long static spans produce many tiny anchor episodes.
        if runs:
            runs = _merge_runs_by_similarity(
                runs, threshold=0.85, max_gap=30,
            )
        n_new = 0
        for run in runs:
            summary = self._make_summary(run)
            self.write_episode(
                t0=run[0].t, t1=run[-1].t,
                summary=summary,
                kind="anchor",  # distinguish from "auto" for diagnostics
                ev_ids=[e.eid for e in run],
                # Slightly lower salience than change-driven episodes so
                # retrieval still prefers the latter when both are
                # equally on-topic.
                salience=1.2 + 0.05 * len(run),
            )
            n_new += 1
        return n_new

    # ---- episodic layer (mid-term) ------------------------------------
    def write_episode(self, t0: int, t1: int, summary: str,
                      kind: str = "episode",
                      ev_ids: Optional[List[int]] = None,
                      salience: float = 1.5,
                      pin_id: Optional[int] = None) -> Dict[str, Any]:
        item = {
            "id": self._next_epid,
            "t0": int(t0), "t1": int(t1),
            "kind": str(kind),
            "summary": str(summary or "")[:6000],
            "ev_ids": list(ev_ids or []),
            "salience": float(salience),
            "pin_id": (int(pin_id) if pin_id is not None else None),
        }
        self._next_epid += 1
        self.episodes.append(item)
        # Back-fill parent_episode_id on the included events.
        if item["ev_ids"]:
            ev_id_set = set(item["ev_ids"])
            for ev in self.events:
                if ev.eid in ev_id_set and ev.parent_episode_id is None:
                    ev.parent_episode_id = item["id"]
        return item

    def auto_aggregate_episodes(self,
                                gap_seconds: int = 10,
                                min_run: int = 5,
                                max_episode_span: int = 45,
                                require_change: bool = True,
                                merge_similar: bool = True,
                                merge_threshold: float = 0.85,
                                merge_max_gap: int = 30) -> int:
        """Aggregate runs of consecutive change/observation events into
        episode summaries. Returns number of episodes newly created.

        Heuristic: walk events sorted by t; start a new run whenever the
        gap to the previous event is > gap_seconds OR the run already
        spans max_episode_span. A run with >= min_run events whose ids
        are not yet covered by any existing episode becomes one episode
        whose summary concatenates the first and last event texts.

        ``require_change``: only emit an episode for a run that contains
        at least one ``is_change=True`` event. Pure static observation
        runs are skipped — this preserves the "store changes" semantics
        at the episodic layer while the working layer still keeps every
        observation verbatim.

        ``merge_similar``: after the gap/span cut, iteratively merge
        adjacent runs whose content signatures are nearly identical
        (mean Jaccard ≥ ``merge_threshold``) and whose inter-run gap
        ≤ ``merge_max_gap``. This collapses long stretches of "model
        repeated the same labels" into single coarse episodes.
        """
        if not self.events:
            return 0
        covered = set()
        for ep in self.episodes:
            covered.update(ep.get("ev_ids") or [])
        evs = sorted(self.events, key=lambda e: e.t)
        runs: List[List[Event]] = []
        cur: List[Event] = []
        for e in evs:
            if e.eid in covered:
                if len(cur) >= min_run:
                    runs.append(cur)
                cur = []
                continue
            if not cur:
                cur = [e]
                continue
            gap = e.t - cur[-1].t
            span = e.t - cur[0].t
            if gap > gap_seconds or span > max_episode_span:
                if len(cur) >= min_run:
                    runs.append(cur)
                cur = [e]
            else:
                cur.append(e)
        if len(cur) >= min_run:
            runs.append(cur)
        if require_change:
            runs = [r for r in runs if any(e.is_change for e in r)]
        if merge_similar and runs:
            runs = _merge_runs_by_similarity(
                runs,
                threshold=float(merge_threshold),
                max_gap=int(merge_max_gap),
            )
        n_new = 0
        for run in runs:
            summary = self._make_summary(run)
            self.write_episode(
                t0=run[0].t, t1=run[-1].t,
                summary=summary,
                kind="auto",
                ev_ids=[e.eid for e in run],
                salience=1.5 + 0.05 * len(run),
            )
            n_new += 1
        return n_new

    # ---- read paths ---------------------------------------------------
    def list(
        self,
        kind: Optional[str] = None,
        t_lo: Optional[int] = None,
        t_hi: Optional[int] = None,
    ) -> List[Event]:
        out = []
        for e in self.events:
            if kind and e.kind != kind:
                continue
            if t_lo is not None and e.t < t_lo:
                continue
            if t_hi is not None and e.t > t_hi:
                continue
            out.append(e)
        return out

    def search(self, query: str, top_k: int = 5,
               t_range: Optional[Tuple[int, int]] = None) -> List[Event]:
        """Keyword-overlap search (no external embedding dep)."""
        q = re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]+", (query or "").lower())
        q_set = set(q)
        scored: List[Tuple[float, Event]] = []
        for e in self.events:
            if e.kind == "note" and str(e.text).lstrip().startswith("[plan]"):
                continue
            if t_range and not (t_range[0] <= e.t <= t_range[1]):
                continue
            toks = set(re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]+", e.text.lower()))
            if not q_set:
                score = e.salience
            else:
                overlap = len(toks & q_set)
                if overlap == 0:
                    continue
                score = overlap * (0.5 + 0.5 * e.salience)
            scored.append((score, e))
        scored.sort(key=lambda x: (-x[0], -x[1].t))
        return [e for _, e in scored[:top_k]]

    # ---- cross-layer retrieval (long_term + episodic + working) -------
    _RETRIEVE_TOK_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")

    @staticmethod
    def _toks(text: str) -> set:
        return set(MemoryStore._RETRIEVE_TOK_RE.findall(
            (text or "").lower()))

    # \u2500\u2500 Hybrid retrieval support \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    # Optional: when a query-embedding function (e.g. vLLM encode) is
    # provided, re-rank keyword results with cosine similarity against
    # pre-computed token-hash event embeddings.

    def set_embed_fn(self, fn: Any) -> None:
        """Register a query \u2192 vector function for hybrid scoring."""
        self._query_embed_fn = fn

    def cache_embedding(self, eid: int, vec: Any) -> None:
        """Cache a token-hash embedding for one event."""
        if hasattr(vec, "tolist"):
            self._event_embeddings[eid] = vec.tolist()
        elif isinstance(vec, list):
            self._event_embeddings[eid] = vec

    def _hybrid_score(self, keyword_score: float, eid: int,
                       query_emb) -> float:
        """Blend BM25-style keyword score with embedding cosine.

        \u03b1 = 0.5 by default \u2014 equal weight to exact match and semantic
        similarity.  Uses an *additive* blend so that semantically
        similar items (cosine > 0) can still be surfaced even when
        keyword overlap is zero (e.g. "poker cards" vs "playing cards").
        The cosine bonus is capped at 0.5 so exact keyword matches
        always dominate.
        """
        if query_emb is None or eid not in self._event_embeddings:
            return keyword_score
        ev = self._event_embeddings[eid]
        if hasattr(query_emb, "tolist"):
            qv = query_emb.tolist() if hasattr(query_emb, "tolist") else query_emb
        else:
            qv = query_emb
        # Cosine similarity.
        dot = sum(a * b for a, b in zip(qv, ev))
        norm_q = (sum(a * a for a in qv) or 1e-8) ** 0.5
        norm_e = (sum(b * b for b in ev) or 1e-8) ** 0.5
        cosine = max(0.0, dot / (norm_q * norm_e))
        alpha = 0.5
        semantic_bonus = (1.0 - alpha) * cosine  # \u2208 [0, 0.5]
        return keyword_score + semantic_bonus

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        t_range: Optional[Tuple[int, int]] = None,
        current_t: Optional[int] = None,
        layers: Tuple[str, ...] = ("long_term", "episodic", "working"),
        change_boost: float = 1.5,
        layer_quota: Optional[Dict[str, int]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Cross-layer keyword-overlap retrieval.

        Score = overlap × salience × layer_prior × change_boost / recency_decay
          - overlap:      |toks(text) ∩ toks(query)|
          - salience:     0.5 + 0.5 * item.salience  (defaults to 0.5)
          - layer_prior:  long_term=1.5, episodic=1.2, working=1.0
          - change_boost: 1.5 if working event has is_change else 1.0
          - recency_decay: 1 + 0.01 * max(0, current_t - t)

        Returns {"long_term": [...], "episodic": [...], "working": [...]}.
        Each item is a dict carrying the relevant fields plus
        ``__score__`` for downstream rendering / debugging.

        ``layer_quota`` is the per-layer cap; defaults distribute
        ``top_k`` as long:ep:work = ceil(top_k*0.25) : ceil(top_k*0.375)
        : remainder, with at least 1 per requested layer when top_k>=3.
        """
        q_set = self._toks(query)
        # ── Query embedding for hybrid scoring ────────────────────────
        query_q_emb = None
        if self._query_embed_fn is not None:
            try:
                query_q_emb = self._query_embed_fn(query)
            except Exception:
                query_q_emb = None
        if not q_set:
            return {l: [] for l in layers}

        layer_priors = {"long_term": 1.5, "episodic": 1.2, "working": 1.0}
        now_t = (int(current_t) if current_t is not None
                 else (self.events[-1].t if self.events else 0))

        def _recency(t: Optional[int]) -> float:
            if t is None:
                return 1.0
            return 1.0 + 0.01 * max(0, now_t - int(t))

        out: Dict[str, List[Tuple[float, Dict[str, Any]]]] = {
            l: [] for l in layers
        }

        # ---- LONG-TERM (pinned_facts) ---------------------------------
        if "long_term" in layers:
            for p in self.pinned_facts:
                if p.get("kind") == "question":
                    continue
                text = p.get("text") or ""
                t = p.get("t")
                if t_range and t is not None and not (
                        t_range[0] <= t <= t_range[1]):
                    continue
                toks = self._toks(text)
                overlap = len(toks & q_set)
                if overlap == 0:
                    continue
                # pinned_facts have no salience field — treat as 1.5 by
                # convention since they are durable model-authored facts.
                sal = 1.5
                raw_score = (overlap * (0.5 + 0.5 * sal)
                         * layer_priors["long_term"]
                         / _recency(t))
                # ── Hybrid: blend with embedding cosine when available
                if self._query_embed_fn is not None and query_q_emb is not None:
                    raw_score = self._hybrid_score(raw_score, -p.get("id", 0), query_q_emb)
                out["long_term"].append((raw_score, {
                    "id": p.get("id"),
                    "kind": p.get("kind"),
                    "t": t,
                    "ep_id": p.get("ep_id"),
                    "text": text,
                    "__score__": round(raw_score, 4),
                }))

        # ---- EPISODIC (episodes) --------------------------------------
        if "episodic" in layers:
            for ep in self.episodes:
                t0 = ep.get("t0"); t1 = ep.get("t1")
                if t_range:
                    if t0 is None or t1 is None:
                        continue
                    # any-overlap with query range
                    if t1 < t_range[0] or t0 > t_range[1]:
                        continue
                summary = ep.get("summary") or ""
                toks = self._toks(summary)
                overlap = len(toks & q_set)
                if overlap == 0:
                    continue
                sal = float(ep.get("salience") or 1.5)
                raw_score = (overlap * (0.5 + 0.5 * sal)
                         * layer_priors["episodic"]
                         / _recency(t1))
                if self._query_embed_fn is not None and query_q_emb is not None:
                    raw_score = self._hybrid_score(
                        raw_score, -ep.get("id", 0) - 100000, query_q_emb)
                out["episodic"].append((raw_score, {
                    "id": ep.get("id"),
                    "kind": ep.get("kind"),
                    "t0": t0, "t1": t1,
                    "ev_ids": ep.get("ev_ids") or [],
                    "pin_id": ep.get("pin_id"),
                    "summary": summary,
                    "__score__": round(raw_score, 4),
                }))

        # ---- WORKING (events) -----------------------------------------
        if "working" in layers:
            for e in self.events:
                if e.kind == "note" and str(e.text).lstrip().startswith("[plan]"):
                    continue
                if t_range and not (t_range[0] <= e.t <= t_range[1]):
                    continue
                # Search both full text and delta snippet.
                toks = self._toks(e.text)
                if e.delta:
                    toks |= self._toks(e.delta)
                overlap = len(toks & q_set)
                if overlap == 0:
                    continue
                sal = float(e.salience or 1.0)
                cb = change_boost if e.is_change else 1.0
                raw_score = (overlap * (0.5 + 0.5 * sal)
                         * layer_priors["working"] * cb
                         / _recency(e.t))
                if self._query_embed_fn is not None and query_q_emb is not None:
                    raw_score = self._hybrid_score(raw_score, e.eid, query_q_emb)
                d = e.to_dict()
                d["__score__"] = round(raw_score, 4)
                out["working"].append((raw_score, d))

        # ---- per-layer quota + sort -----------------------------------
        if layer_quota is None:
            # default: long:ep:work = 0.25 : 0.375 : 0.375 of top_k
            ql = max(1, int(round(top_k * 0.25))) if top_k >= 3 else max(1, top_k // 3)
            qe = max(1, int(round(top_k * 0.375))) if top_k >= 3 else max(1, top_k // 3)
            qw = max(1, top_k - ql - qe)
            layer_quota = {"long_term": ql, "episodic": qe, "working": qw}

        result: Dict[str, List[Dict[str, Any]]] = {}
        for l in layers:
            scored = sorted(out[l], key=lambda x: (-x[0], 0))
            cap = layer_quota.get(l, top_k)
            result[l] = [item for _, item in scored[:cap]]

        # If under-filled, top up working layer with remainder budget.
        used = sum(len(v) for v in result.values())
        if used < top_k and "working" in layers:
            extra_budget = top_k - used
            taken_ids = {item["id"] for item in result.get("working", [])}
            extras_scored = sorted(out["working"], key=lambda x: (-x[0], 0))
            for _, item in extras_scored:
                if extra_budget <= 0:
                    break
                if item["id"] in taken_ids:
                    continue
                result["working"].append(item)
                taken_ids.add(item["id"])
                extra_budget -= 1
        return result

    # ---- prompt brief --------------------------------------------------
    def snapshot(
        self,
        token_budget: int = 3000,
        max_events: int = 30,
        current_t: Optional[int] = None,
        head_seconds: int = 25,
        tail_seconds: int = 20,
        head_max: int = 10,
        tail_max: int = 24,
        summary_max: int = 30,
        max_episodes: Optional[int] = None,
        auto_episodes: bool = True,
        mode: str = "continuous",
        recall_query: Optional[str] = None,
        recall_topk: int = 8,
        selective_head_max: int = 6,
        selective_tail_max: int = 8,
    ) -> Dict[str, Any]:
        """Render a hierarchical MEMORY brief.

        Three layers (rendered top→bottom):
          LONG-TERM (pinned_facts): durable facts, never evicted.
          EPISODIC  (episodes)    : aggregated [t0,t1] event summaries
                                    spanning multiple seconds.
          WORKING   (events)      : second-level head/middle/tail events.

        Working layer is itself split into head (anchor) / middle / tail
        (recent) regions for rendering only.

        Modes (controls only the WORKING layer; long_term + episodic
        always render in full):

        * ``mode='full'``:      head/middle/tail all verbatim.
        * ``mode='continuous'`` (default, recommended):
            - head: ALL events with ``e.t < head_seconds`` verbatim.
            - tail: ALL events with ``e.t >= now_t - tail_seconds`` verbatim.
            - middle: NOT rendered verbatim — replaced by an aggregation
              note pointing to the episodic layer; optional retrieval
              recall (``recall_query``) brings back top-K relevant
              middle events.
        * ``mode='selective'``: legacy mode kept for backward compat
            (caps head to ``selective_head_max``, tail to
            ``selective_tail_max``, replaces middle with retrieval
            recall). Discouraged because it silently drops middle.

        ``max_episodes``: when ``None`` (default), every episode is
        rendered. Setting an integer caps the rendered count for
        debugging only.

        ``head_max`` / ``tail_max`` / ``summary_max`` / ``token_budget``
        are kept in the signature for API compatibility but are NOT
        enforced in ``mode='full'`` or ``mode='continuous'``.
        """
        if not (self.events or self.entities or self.tracks
                or self.audio_segments or self.pinned_facts
                or self.episodes):
            return {
                "text": "MEMORY: (empty)",
                "compressed": False,
                "head_t_max": None,
                "tail_t_min": None,
                "n_pinned": 0,
                "n_episodes": 0,
                "n_head": 0,
                "n_middle": 0,
                "n_tail": 0,
                "n_dropped": 0,
            }

        now_t = current_t if current_t is not None else (
            self.events[-1].t if self.events else 0)

        # ---- auto build episodic layer (mid-term) -------------------
        if auto_episodes:
            self.auto_aggregate_episodes()
            # Time-coverage fallback: even when ``auto_aggregate_episodes``
            # finds nothing (e.g. the model wrote `delta=(no change)`
            # for a long span, so no run carries an ``is_change`` flag),
            # we still want the episodic layer to extend close to
            # ``now_t`` so retrieval/rendering does not blind-spot the
            # most recent half of the video. ``ensure_episodic_coverage``
            # emits an "anchor"-kind episode covering any gap > 30s
            # that ends before the working layer's verbatim tail.
            self.ensure_episodic_coverage(
                current_t=now_t,
                max_gap_seconds=30,
                tail_protect=max(8, int(tail_seconds)),
                min_run=2,
                max_episode_span=45,
            )

        # ---- partition events into head / tail / middle --------------
        # Partitioning is for *rendering only* — we keep every event in
        # every section, no salience-based selection, no count cap.
        head_cutoff = head_seconds
        tail_cutoff = max(0, now_t - tail_seconds)
        # If head and tail windows would overlap (short videos / early
        # frames), shrink the tail to start right after the head so we
        # don't double-render frames.
        if tail_cutoff < head_cutoff:
            tail_cutoff = head_cutoff

        head_evs: List[Event] = []
        tail_evs: List[Event] = []
        middle_evs: List[Event] = []
        for e in self.events:
            if e.t < head_cutoff:
                head_evs.append(e)
            elif e.t >= tail_cutoff:
                tail_evs.append(e)
            else:
                middle_evs.append(e)

        head_evs.sort(key=lambda e: e.t)
        tail_evs.sort(key=lambda e: e.t)
        middle_evs.sort(key=lambda e: e.t)

        # Working-layer rendering policy.
        recall_kept: List[Event] = []
        ranked_middle: List[Event] = []
        n_dropped_middle = 0
        if mode == "selective":
            head_kept = head_evs[:max(0, int(selective_head_max))]
            tail_kept = (tail_evs[-max(0, int(selective_tail_max)):]
                         if selective_tail_max > 0 else [])
            taken_ids = {e.eid for e in head_kept} | {e.eid for e in tail_kept}
            if recall_query and recall_topk > 0:
                try:
                    hits = self.retrieve(
                        query=str(recall_query),
                        top_k=int(recall_topk),
                        current_t=now_t,
                        layers=("working",),
                        layer_quota={"working": int(recall_topk)},
                    )
                    wk = (hits or {}).get("working") or []
                    id_to_ev = {e.eid: e for e in self.events}
                    for it in wk:
                        eid = it.get("id")
                        if eid in taken_ids:
                            continue
                        ev = id_to_ev.get(eid)
                        if ev is None:
                            continue
                        recall_kept.append(ev)
                        taken_ids.add(eid)
                except Exception:
                    recall_kept = []
            recall_kept.sort(key=lambda e: e.t)
            n_dropped_middle = max(
                0, len(middle_evs) - len(recall_kept)
            )
        elif mode == "continuous":
            # Head + tail are kept VERBATIM in full (no count cap); the
            # middle is replaced by an aggregation note pointing to
            # episodic. Optional recall brings back the top-K middle
            # events for the question at hand (when ``recall_query``
            # is set).
            head_kept = head_evs
            tail_kept = tail_evs
            taken_ids = {e.eid for e in head_kept} | {e.eid for e in tail_kept}
            if recall_query and recall_topk > 0:
                try:
                    hits = self.retrieve(
                        query=str(recall_query),
                        top_k=int(recall_topk),
                        current_t=now_t,
                        layers=("working",),
                        layer_quota={"working": int(recall_topk)},
                    )
                    wk = (hits or {}).get("working") or []
                    id_to_ev = {e.eid: e for e in self.events}
                    for it in wk:
                        eid = it.get("id")
                        if eid in taken_ids:
                            continue
                        ev = id_to_ev.get(eid)
                        if ev is None:
                            continue
                        # Only surface middle-region events; head/tail
                        # are already verbatim.
                        if ev.t < head_cutoff or ev.t >= tail_cutoff:
                            continue
                        recall_kept.append(ev)
                        taken_ids.add(eid)
                except Exception:
                    recall_kept = []
            recall_kept.sort(key=lambda e: e.t)
            # ``n_dropped_middle`` counts middle events not rendered
            # verbatim. They are NOT lost: episodic captures them.
            n_dropped_middle = max(
                0, len(middle_evs) - len(recall_kept)
            )
        else:
            # mode == "full": no truncation anywhere.
            head_kept = head_evs
            tail_kept = tail_evs
            ranked_middle = middle_evs
            n_dropped_middle = 0

        # ---- render ---------------------------------------------------
        n_total = len(self.events)
        lines: List[str] = [
            f"MEMORY ({n_total} total events in store; rendering subset below):"
        ]

        # LONG-TERM
        if self.pinned_facts:
            active_pins = [p for p in self.pinned_facts
                           if p.get("superseded_by") is None]
            archived_pins = [p for p in self.pinned_facts
                             if p.get("superseded_by") is not None]
            if active_pins:
                lines.append("  long_term (pinned, durable):")
                for p in active_pins:
                    t_tag = (f"t={p['t']:>4d} "
                             if p.get("t") is not None else "")
                    subj_tag = (f" [subj={p['subject']}]"
                                if p.get("subject") else "")
                    ep_ptr = (f" ↘ep#{p['ep_id']}"
                              if p.get("ep_id") is not None else "")
                    lines.append(
                        f"    [#{p['id']}] {t_tag}[{p['kind'][:3]}]"
                        f"{subj_tag}{ep_ptr} {p['text']}"
                    )
            if archived_pins:
                lines.append(
                    "  long_term archived (superseded — DO NOT trust as "
                    "current state; kept for history. Use memory_search "
                    "if you need to verify the timeline of changes):"
                )
                for p in archived_pins:
                    t_tag = (f"t={p['t']:>4d} "
                             if p.get("t") is not None else "")
                    sup_tag = (f" →#{p['superseded_by']}"
                               f"@t={p.get('superseded_at')}")
                    subj_tag = (f" [subj={p['subject']}]"
                                if p.get("subject") else "")
                    ep_ptr = (f" ↘ep#{p['ep_id']}"
                              if p.get("ep_id") is not None else "")
                    lines.append(
                        f"    [#{p['id']}] {t_tag}[{p['kind'][:3]}]"
                        f"{subj_tag}{ep_ptr}{sup_tag} {p['text']}"
                    )

        # EPISODIC (aggregated [t0,t1] runs)
        rendered_eps: List[Dict[str, Any]] = []
        if self.episodes:
            eps_chrono = sorted(self.episodes, key=lambda x: x["t0"])
            if max_episodes is None or len(eps_chrono) <= int(max_episodes):
                rendered_eps = eps_chrono
            else:
                # Cap by salience (highest first), but render in time order.
                eps_by_sal = sorted(
                    self.episodes,
                    key=lambda x: (-x.get("salience", 1.0), -x["t1"]))
                rendered_eps = sorted(
                    eps_by_sal[:int(max_episodes)], key=lambda x: x["t0"])
            if rendered_eps:
                lines.append(
                    f"  episodic (kept {len(rendered_eps)}/{len(self.episodes)}):"
                )
                for ep in rendered_eps:
                    summary = (ep.get("summary") or "").strip()
                    summary_lines = summary.split("\n")
                    head = summary_lines[0] if summary_lines else ""
                    # Cross-layer pointer markers so the model can trace
                    # evidence: ev_ids shows which working events back
                    # this episode; pin_id links to durable long-term pin.
                    ev_ids = ep.get("ev_ids") or []
                    pin_id = ep.get("pin_id")
                    ev_tag = ""
                    if ev_ids:
                        if len(ev_ids) <= 4:
                            ev_tag = f" → ev#{','.join(str(i) for i in ev_ids)}"
                        else:
                            ev_tag = (f" → ev#{ev_ids[0]}-ev#{ev_ids[-1]}"
                                      f" ({len(ev_ids)} events)")
                    pin_tag = f" → pin#{pin_id}" if pin_id else ""
                    lines.append(
                        f"    [ep#{ep['id']}] t={ep['t0']}-{ep['t1']}s "
                        f"[{ep['kind'][:3]}]{ev_tag}{pin_tag} {head}"
                    )
                    # Indent subsequent lines so they nest visually
                    # under the [ep#…] header. The downstream prompt
                    # rendering picks them up as part of the same
                    # episodic block.
                    for tail_line in summary_lines[1:]:
                        if tail_line.strip():
                            lines.append(f"      {tail_line}")

        # Working-layer side indices: entities / tracks / audio.
        # No truncation — these are typically small.
        if self.entities:
            ents = ", ".join(
                f"{eid}({v.get('label','?')})"
                for eid, v in self.entities.items()
            )
            lines.append(f"  entities: {ents}")
        if self.tracks:
            for tid, tr in self.tracks.items():
                tlen = len(tr.get("traj", []))
                summ = tr.get("summary", "")
                lines.append(f"  track[{tid}] traj={tlen} {summ}".rstrip())
        if self.audio_segments:
            for a in self.audio_segments:
                lines.append(
                    f"  audio[{a['t0']}-{a['t1']}s]: {a['text'][:60]}"
                )

        def _fmt_event(e, recent=False):
            if e.is_change:
                kind_tag = "Δ"
            elif e.kind == "change":
                kind_tag = "Δ"
            else:
                kind_tag = e.kind[:3]
            # Cross-layer pointer: show which episode this event belongs to.
            ep_ptr = ""
            if getattr(e, "parent_episode_id", None) is not None:
                ep_ptr = f" →ep#{e.parent_episode_id}"
            # Recency marker: tail events are the most recent observations
            # before the question — flag them so the model trusts WHAT JUST
            # HAPPENED over high-frequency older evidence.
            recent_tag = " [RECENT, HIGHEST PRIORITY]" if recent else ""
            return (f"    t={e.t:>4d} [{kind_tag}]{ep_ptr}{recent_tag} "
                    f"{e.text}")

        # WORKING layer: head / middle / tail second-level events
        if head_kept or ranked_middle or tail_kept or recall_kept:
            mode_tag = f" ({mode})" if mode != "full" else ""
            lines.append(f"  working (second-level){mode_tag}:")
        if head_kept:
            lines.append(
                f"    head (anchor t<{head_cutoff}s, "
                f"{len(head_kept)} events, verbatim):"
            )
            for e in head_kept:
                lines.append(_fmt_event(e))
        if mode in ("selective", "continuous") and len(middle_evs) > 0:
            # Tell the model that the middle region is NOT silently
            # dropped — it lives in the episodic layer and is
            # retrievable. This is critical for the "no-deletions"
            # guarantee: middle events are aggregated, not lost.
            mid_lo = middle_evs[0].t
            mid_hi = middle_evs[-1].t
            ep_range = ""
            if rendered_eps:
                covering = [ep for ep in rendered_eps
                            if ep["t1"] >= mid_lo and ep["t0"] <= mid_hi]
                if covering:
                    ep_range = (f" — see episodic "
                                f"[ep#{covering[0]['id']}..ep#"
                                f"{covering[-1]['id']}]")
            # ── Content digest: surface key entities / actions / change
            # timestamps so the model knows what's in the middle without
            # reading every verbatim event.  Acts as a "table of contents"
            # for memory_search / replay lookups.
            mid_entities: set = set()
            mid_actions: set = set()
            change_ts: List[int] = []
            for e in middle_evs:
                txt = getattr(e, "text", "") or ""
                fields = _split_fields(txt)
                for k in ("actor", "object"):
                    v = fields.get(k, "")
                    if v:
                        for tok in re.findall(r"[A-Za-z一-鿿]{3,}", v):
                            mid_entities.add(tok.lower())
                act = fields.get("action", "")
                if act:
                    for tok in re.findall(r"[A-Za-z]{4,}", act):
                        mid_actions.add(tok.lower())
                if getattr(e, "is_change", False):
                    change_ts.append(e.t)
            digest_parts: List[str] = []
            if mid_entities:
                ents_sorted = sorted(mid_entities,
                                     key=lambda x: -len(x))[:15]
                digest_parts.append(f"entities: {', '.join(ents_sorted)}")
            if mid_actions:
                acts_sorted = sorted(mid_actions,
                                     key=lambda x: -len(x))[:10]
                digest_parts.append(f"actions: {', '.join(acts_sorted)}")
            if change_ts:
                ts_str = ", ".join(f"t={t}" for t in change_ts[:10])
                digest_parts.append(f"change_frames: {ts_str}")
            digest = (" | ".join(digest_parts)) if digest_parts else ""
            if digest:
                digest = f"  digest: {digest}"
            lines.append(
                f"    middle (t={mid_lo}-{mid_hi}s, {len(middle_evs)} "
                f"events aggregated into episodic above{ep_range}; "
                f"call memory_search/replay to recall verbatim)."
                f"{digest}"
            )
        if mode in ("selective", "continuous") and recall_kept:
            lines.append(
                f"    [recall] top-{len(recall_kept)} relevant from middle "
                f"(query=\"{(recall_query or '')[:60]}\"):"
            )
            for e in recall_kept:
                lines.append(_fmt_event(e))
        if ranked_middle:
            mid_lo = ranked_middle[0].t
            mid_hi = ranked_middle[-1].t
            lines.append(
                f"    middle (verbatim t={mid_lo}-{mid_hi}s, "
                f"{len(ranked_middle)} events):"
            )
            for e in ranked_middle:
                lines.append(_fmt_event(e))
        if tail_kept:
            lines.append(
                f"    tail (recent t>={tail_cutoff}s, "
                f"{len(tail_kept)} events, verbatim — MOST RECENT, "
                f"HIGHEST PRIORITY: trust these over older/repeated "
                f"evidence):"
            )
            for e in tail_kept:
                lines.append(_fmt_event(e, recent=True))

        text = "\n".join(lines)

        # ── Soft-budget guard (DeltaStreamer O(1) prompt protection) ────
        # Continuous mode renders head + tail verbatim. When observations
        # are verbose (O1 detailed fields) on very long videos the rendered
        # MEMORY can grow until system+memory+image tokens exceed the vLLM
        # ``max_model_len`` and the engine aborts the whole request. To keep
        # the prompt bounded we drop the OLDEST verbatim working-event lines
        # once the text exceeds a generous hard cap. Pinned facts, episodic
        # summaries, section headers, the head anchor and the most-recent
        # tail events are always kept. This only triggers on pathological
        # growth, so normal samples are unaffected.
        soft_trimmed = 0
        hard_cap_chars = int(token_budget) * 12 if token_budget and token_budget > 0 else 0
        # Never trim below a safety floor so short/normal samples are intact.
        if hard_cap_chars and hard_cap_chars < 60000:
            hard_cap_chars = 60000
        if hard_cap_chars and len(text) > hard_cap_chars:
            ev_idx = [i for i, l in enumerate(lines)
                      if re.match(r"\s+t=\s*\d+ \[", l)]
            if ev_idx:
                protect_head = set(ev_idx[:max(1, head_max)])
                protect_tail = set(ev_idx[-max(1, tail_max // 2):])
                trimmable = [i for i in ev_idx
                             if i not in protect_head and i not in protect_tail]
                drop: set = set()
                cur_len = len(text)
                for i in trimmable:
                    if cur_len <= hard_cap_chars:
                        break
                    cur_len -= len(lines[i]) + 1
                    drop.add(i)
                if drop:
                    lines = [l for k, l in enumerate(lines) if k not in drop]
                    lines.append(
                        f"    [soft-trimmed {len(drop)} oldest verbatim "
                        f"event(s) to fit context; recall via "
                        f"memory_search/replay]."
                    )
                    text = "\n".join(lines)
                    soft_trimmed = len(drop)

        # ``token_budget`` is otherwise kept for API compatibility; the
        # guard above is the only enforcement in continuous/full modes.

        compressed = (mode in ("selective", "continuous")
                      and (n_dropped_middle > 0 or soft_trimmed > 0))
        return {
            "text": text,
            "compressed": compressed,
            "head_t_max": (head_kept[-1].t if head_kept else None),
            "tail_t_min": (tail_kept[0].t if tail_kept else None),
            "n_pinned": len(self.pinned_facts),
            "n_episodes": len(rendered_eps),
            "n_head": len(head_kept),
            "n_middle": len(ranked_middle),
            "n_tail": len(tail_kept),
            "n_recall": len(recall_kept),
            "mode": mode,
            "n_dropped": n_dropped_middle
                        + max(0, len(head_evs) - len(head_kept))
                        + max(0, len(tail_evs) - len(tail_kept)),
        }


# ---------------------------------------------------------------------------
# Rule-based Summarizer (Phase-A; can be swapped to LLM later)
# ---------------------------------------------------------------------------
_SILENCE_TOKENS = ("</Silence>", "</Standby>")
_RESPONSE_TOKEN = "</Response>"


class RuleSummarizer:
    """Compress a single-round model response into an Event payload.

    Working-layer policy: keep the FULL ``<Observation>`` body (task /
    scene / actions / objects / spatial / delta) and additionally
    extract the ``delta=...`` snippet so the working layer can flag
    which frames represent a *change*. The ``is_change`` / ``delta``
    fields propagate to retrieval, where change frames receive a
    salience boost.

    Rules:
    * Pure ``</Silence>`` / ``</Standby>`` → ``None`` (no event).
    * ``<Observation>(no change)</Observation>`` (whole body) → ``None``.
    * ``<Observation>...</Observation>`` → keep full body. If a
      ``delta=...`` field is found AND not "(no change)", set
      ``is_change=True``, ``delta=<snippet>`` and ``salience=1.5``;
      otherwise ``salience=1.0``.
    * ``</Response>`` → record final answer (kind=qa, salience=2.0).
    * Otherwise → store remainder verbatim as observation.
    """

    _TOOL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.S)
    _OBS_RE = re.compile(r"<Observation>(.*?)</Observation>", re.S | re.I)
    _NOCHANGE_RE = re.compile(
        r"^\s*\(?\s*no[\s_-]*change\s*\)?\s*\.?\s*$", re.I)
    # Extract delta= field (change description) from observation text.
    _DELTA_RE = re.compile(r"delta\s*[:=]\s*(.+?)(?:\n|$)", re.S | re.I)
    # Strip <think>...</think> chain-of-thought blocks before any other
    # parsing. Without this, when a model (e.g. Qwen3.5 with thinking
    # enabled) omits the <Observation>...</Observation> wrapper, the
    # fallback "store remainder verbatim" branch would dump the entire
    # reasoning trace into MEMORY, and the </Response> branch could
    # mistake a literally-quoted "</Response>" inside <think> for the
    # final answer.
    _THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.S | re.I)
    # Tolerate an unterminated trailing <think> block (response cut off
    # by max_tokens mid-reasoning).
    _THINK_OPEN_TAIL_RE = re.compile(r"<think\b[^>]*>.*\Z", re.S | re.I)

    def __init__(self, dedup_jaccard: float = 0.85,
                 delta_repeat_window: int = 8,
                 delta_repeat_threshold: int = 2,
                 novelty_window: int = 12,
                 novelty_threshold: float = 0.3) -> None:
        # ``dedup_jaccard`` kept for backward-compatible ctor signature
        # but no longer used — deduplication is disabled by request.
        self.dedup_jaccard = float(dedup_jaccard)
        # Anti-template delta detection: when the current frame's delta
        # phrase has already been emitted ``delta_repeat_threshold`` or
        # more times within the last ``delta_repeat_window`` deltas, we
        # treat this frame as a "template-copy" — the model is just
        # repeating a phrase rather than describing a new change. Such
        # frames are demoted to ``is_change=False, salience=0.5`` so the
        # episodic aggregator does not split a fresh run on them and so
        # retrieval stops boosting copy-pasted boilerplate.
        self._recent_deltas: Deque[str] = deque(
            maxlen=max(2, int(delta_repeat_window)))
        self._delta_repeat_threshold = max(1, int(delta_repeat_threshold))
        # ── Pillar 1: Information Novelty Scoring ──────────────────────
        # Maintain a sliding window of recent observation field-parses so
        # ``compute_novelty`` can measure how much *new* information the
        # current frame contributes relative to recent history.
        # Fields below ``novelty_threshold`` receive a salience demotion
        # (they are near-duplicates of recent observations); fields above
        # it are boosted and treated as change-bearing.
        self._recent_fields: Deque[str] = deque(
            maxlen=max(4, int(novelty_window)))
        self._novelty_threshold = float(novelty_threshold)
        # ── Pillar 1: Entity extraction ─────────────────────────────────
        # Simple regex to extract entity names from observation text
        # (actors + named objects). Used by MemoryGraph for entity indexing.
        self._ENTITY_EXTRACT_RE = re.compile(
            r'(?:actors?|objects?)\s*[:=]\s*(.+?)(?:\n|;(?:\s*(?:objects?|'
            r'actions?|ocr|scene|delta)\s*[:=])|\Z)',
            re.IGNORECASE | re.S,
        )

    @staticmethod
    def _norm_delta(d: str) -> str:
        """Normalise a delta string for repetition comparison.

        Lower-case, collapse whitespace, strip trailing punctuation.
        """
        if not d:
            return ""
        s = re.sub(r"\s+", " ", d.strip().lower())
        s = re.sub(r"[\.;,!?\s]+$", "", s)
        return s

    # ── Pillar 1: Information Novelty Scoring ──────────────────────────
    # Weights for field-weighted novelty: changes (delta) and actions are
    # the most discriminative signals for whether a frame carries *new*
    # information; scene type and OCR change less often and contribute
    # less to the novelty signal.
    _NOVELTY_FIELD_WEIGHTS = {
        "delta": 0.35, "action": 0.25, "actor": 0.15,
        "object": 0.10, "ocr": 0.10, "scene": 0.05,
    }

    # Tokeniser used by ``compute_novelty`` for field-level Jaccard.
    _NOVELTY_TOK_RE = re.compile(r"[A-Za-z0-9_一-鿿]{2,}")

    def compute_novelty(self, observation_text: str) -> float:
        """Field-weighted information novelty score ∈ [0, 1].

        Parses *observation_text* into labelled fields via
        :func:`_split_fields`, then for each field computes the maximum
        Jaccard token-set similarity against the same field in every
        entry of ``self._recent_fields``.  The per-field novelty is
        ``1 - max_similarity``.  The overall score is a weighted sum
        across fields using :attr:`_NOVELTY_FIELD_WEIGHTS`.

        Returns 1.0 when ``_recent_fields`` is still empty (cold-start).
        """
        if not observation_text:
            return 0.0
        fields = _split_fields(observation_text)
        if not fields:
            return 0.0

        # Tokenise each present field once.
        field_toks: Dict[str, set] = {}
        for key, val in fields.items():
            toks = set(t.lower() for t
                       in self._NOVELTY_TOK_RE.findall(val or ""))
            if toks:
                field_toks[key] = toks

        if not field_toks:
            return 0.0

        # Cold-start: no history → maximum novelty.
        # Must still store fields so subsequent calls see history.
        if not self._recent_fields:
            self._recent_fields.append(fields)
            return 1.0

        # Per-field max similarity against the recent window.
        weighted_sum = 0.0
        total_weight = 0.0
        for key, cur_toks in field_toks.items():
            w = self._NOVELTY_FIELD_WEIGHTS.get(key, 0.05)
            total_weight += w
            max_sim = 0.0
            for hist_fields in self._recent_fields:
                hist_val = hist_fields.get(key)
                if not hist_val:
                    continue
                hist_toks = set(t.lower() for t
                                in self._NOVELTY_TOK_RE.findall(hist_val))
                if not hist_toks:
                    continue
                inter = len(cur_toks & hist_toks)
                union = len(cur_toks | hist_toks)
                sim = inter / union if union > 0 else 0.0
                if sim > max_sim:
                    max_sim = sim
            weighted_sum += w * (1.0 - max_sim)

        # Store the parsed fields for future novelty queries.
        self._recent_fields.append(fields)

        if total_weight <= 0:
            return 1.0
        return weighted_sum / total_weight

    _ENT_NAME_RE = re.compile(
        r'(?:[A-Z][a-z]+(?:\s+(?:in|with|wearing|holding)\s+[a-z]+'
        r'(?:\s+[a-z]+)*)?)'
        r'|(?:\b(?:man|woman|person|child|dog|cat|bird|car|cup|bottle|'
        r'knife|book|ball|chair|table|door|window|phone|bag|hat|shoe'
        r'|glove|box|plate|bowl|spoon|fork|laptop|screen|sign|clock'
        r'|lamp|plant|flower|tree|rock|cloud|water|fire|smoke'
        r'|chef|cook|waiter|waitress|player|driver|worker|student|teacher'
        r'|doctor|nurse|officer|guard|customer|shopper|clerk|mechanic'
        r'|artist|musician|dancer|athlete|referee|coach|spectator'
        r'|passenger|pedestrian|cyclist|rider|pilot|captain|sailor'
        r'|farmer|gardener|painter|plumber|electrician|carpenter|baker'
        r'|butcher|barber|tailor|cleaner|visitor|guest|host|presenter'
        r'|performer|actor|actress|singer|speaker|listener|viewer'
        r'|eggplant|pepper|carrot|onion|potato|tomato|cucumber|lettuce'
        r'|cabbage|broccoli|spinach|celery|radish|mushroom|zucchini'
        r'|squash|pumpkin|gourd|melon|apple|orange|banana|grape|lemon'
        r'|lime|peach|pear|plum|cherry|berry|strawberry|blueberry'
        r'|pineapple|mango|coconut|avocado|nut|bean|rice|pasta|bread'
        r'|cheese|meat|fish|chicken|beef|pork|egg|milk|butter|oil'
        r'|salt|sugar|flour|sauce|soup|salad|sandwich|burger|pizza'
        r'|taco|burrito|sushi|noodle|ramen|curry|stew|roast|grill'
        r'|fry|bake|boil|steam|simmer|saute|stir|whisk|knead|roll'
        r'|chop|slice|dice|mince|grate|peel|mash|puree|blend|mix'
        r'|pour|scoop|ladle|spread|drizzle|sprinkle|garnish|season'
        r'|counter|stove|oven|fridge|freezer|sink|cabinet|drawer|shelf'
        r'|pan|pot|skillet|wok|grill|toaster|microwave|blender|mixer'
        r'|whisk|spatula|ladle|tongs|peeler|grater|strainer|colander'
        r'|cutting_board|chopping_board|measuring_cup|mixing_bowl)\b)',
        re.IGNORECASE,
    )

    @classmethod
    def extract_entities(cls, observation_text: str) -> List[str]:
        """Best-effort entity extraction from observation text.

        Scans the ``actors`` and ``objects`` fields for noun phrases
        that could serve as entity identifiers. Returns a deduplicated
        list of lower-case entity slugs.
        """
        fields = _split_fields(observation_text or "")
        candidates: List[str] = []
        for key in ("actor", "object"):
            val = fields.get(key, "")
            if val:
                candidates.append(val)
        if not candidates:
            # Fallback: scan the first 300 chars for capitalised nouns.
            head = (observation_text or "")[:300]
            for m in cls._ENT_NAME_RE.finditer(head):
                slug = m.group(0).strip().lower()
                slug = re.sub(r"\s+", "_", slug)
                if len(slug) >= 3:
                    candidates.append(slug)
            return list(dict.fromkeys(candidates))[:8]  # dedup, cap

        # Parse each field's content for named entities.
        # Strategy: extract role/noun words from numbered-list entries
        # like "1) chef in white ..." or bullet entries.
        entities: List[str] = []
        # 1) Numbered-list extraction: "1) chef in white jacket"
        _NUM_ITEM_RE = re.compile(
            r'(?:\d+[\)\.]\s*)'                    # "1) " or "1. "
            r'([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+)*)',  # "chef" or "sous chef"
        )
        # 2) Role-noun extraction: "a chef", "the waiter", "a person"
        _ROLE_RE = re.compile(
            r'\b(?:a|the|an)\s+([a-z]{3,}(?:\s+[a-z]{3,}){0,2})'
            r'\s+(?:in|with|wearing|holding|at|standing|sitting|moving|'
            r'walking|cooking|chopping|washing|carrying|pushing|pulling)',
            re.IGNORECASE,
        )
        for chunk in candidates:
            # Numbered items.
            for m in _NUM_ITEM_RE.finditer(chunk):
                slug = m.group(1).strip().lower().replace(" ", "_")
                if len(slug) >= 3:
                    entities.append(slug)
            # Role-noun phrases.
            for m in _ROLE_RE.finditer(chunk):
                slug = m.group(1).strip().lower().replace(" ", "_")
                if len(slug) >= 3:
                    entities.append(slug)
            # Common-noun fallback via the class-level regex.
            for m in cls._ENT_NAME_RE.finditer(chunk):
                slug = m.group(0).strip().lower()
                slug = re.sub(r"\s+", "_", slug)
                if len(slug) >= 3:
                    entities.append(slug)
        return list(dict.fromkeys(entities))[:12]  # dedup, cap

    def __call__(self, response: str, t: int,
                 ref_frame: Optional[int] = None) -> Optional[Dict[str, Any]]:
        if not response:
            return None
        s = response.strip()
        if s in _SILENCE_TOKENS or s == "":
            return None
        # Drop chain-of-thought BEFORE any other parsing so that:
        #   (a) the </Response> branch never picks up a literal
        #       "</Response>" token that the model quoted inside <think>;
        #   (b) the verbatim-fallback branch at the bottom doesn't dump
        #       the whole reasoning trace into MEMORY when the model
        #       skips the <Observation>...</Observation> wrapper.
        s = self._THINK_RE.sub("", s)
        s = self._THINK_OPEN_TAIL_RE.sub("", s).strip()
        if not s or s in _SILENCE_TOKENS:
            return None
        # Drop tool-call blocks for memory
        s = self._TOOL_RE.sub("", s).strip()
        if not s:
            return None
        # Post-hoc observation block — preferred path before question_time.
        m = self._OBS_RE.search(s)
        if m:
            obs_full = m.group(1).strip()
            if not obs_full:
                return None
            # Pure "(no change)" markers: under the new memory
            # semantics the model is forbidden from emitting these,
            # but if it slips through we still record a low-salience
            # static event so the episodic layer can keep full time
            # coverage. The body is rewritten to a neutral placeholder
            # to avoid polluting retrieval with the literal "(no
            # change)" string.
            if self._NOCHANGE_RE.match(obs_full):
                return {
                    "kind": "observation",
                    "text": "(static frame)",
                    "delta": None,
                    "is_change": False,
                    "ref_frame": ref_frame,
                    "salience": 0.5,
                }
            # Extract delta=... if present (and meaningful).
            delta_text: Optional[str] = None
            is_change = False
            dm = self._DELTA_RE.search(obs_full)
            if dm:
                d = dm.group(1).strip()[:240]
                if d and not self._NOCHANGE_RE.match(d):
                    delta_text = d
                    is_change = True
            # Anti-template repetition detection. We only consult the
            # window when a delta is actually present; otherwise the
            # frame is already non-change and the window is unaffected.
            repeated = False
            if delta_text:
                norm = self._norm_delta(delta_text)
                # count prior occurrences of this normalised delta
                # already in the window (PRIOR to inserting the new one).
                prior = sum(1 for x in self._recent_deltas if x == norm)
                if prior >= self._delta_repeat_threshold:
                    repeated = True
                self._recent_deltas.append(norm)
            if repeated:
                # Demote the event: same body so the working layer still
                # shows what the model said, but the episodic aggregator
                # / retrieval treat it as static. We tag the body with a
                # marker so post-mortem inspection can spot template
                # copies easily.
                nov = self.compute_novelty(obs_full)
                entities = self.extract_entities(obs_full)
                return {
                    "kind": "observation",
                    "text": f"[template-copy | nov={nov:.2f}] {obs_full}",
                    "delta": delta_text,
                    "is_change": False,
                    "ref_frame": ref_frame,
                    "salience": 0.5,
                    "entities": entities,
                }
            # ── Pillar 1: Novelty-adjusted salience ─────────────────
            # High-novelty frames receive a salience boost so episodic
            # aggregation and retrieval prefer them; low-novelty frames
            # (near-duplicates of recent history) are demoted.
            nov = self.compute_novelty(obs_full)
            entities = self.extract_entities(obs_full)
            if nov >= self._novelty_threshold:
                # Genuinely new information: boost.
                adj_salience = 1.0 + 0.5 * nov  # range [1.15, 1.5]
                adj_is_change = True
                prefix = f"[nov={nov:.2f}] "
            else:
                # Low novelty: this frame is nearly identical to recent
                # history. Demote so the episodic aggregator does not
                # split a spurious run on boilerplate observations.
                adj_salience = 0.5
                adj_is_change = False
                prefix = f"[nov={nov:.2f}|static] "
            return {
                "kind": "observation",
                "text": prefix + obs_full,
                "delta": delta_text,
                "is_change": adj_is_change,
                "ref_frame": ref_frame,
                "salience": adj_salience,
                "entities": entities,
            }
        if _RESPONSE_TOKEN in s:
            payload = s.split(_RESPONSE_TOKEN, 1)[-1].strip()[:160]
            return {
                "kind": "qa", "text": f"final: {payload}",
                "ref_frame": ref_frame, "salience": 2.0,
            }
        # plain observation — keep full body (still capped by write_event).
        return {
            "kind": "observation",
            "text": s,
            "ref_frame": ref_frame,
            "salience": 1.0,
        }


# ---------------------------------------------------------------------------
# Tools (the model can call these explicitly)
# ---------------------------------------------------------------------------
def _require_memory(ctx: ToolContext) -> Optional[ToolResult]:
    if ctx.memory is None:
        return ToolResult("memory", False, error="memory not enabled in this run")
    return None


class MemoryWriteTool(Tool):
    name = "memory_write"
    description = (
        "Append a fact to long-term memory. Use this to record key "
        "observations (object appearance, action onset, dialogue gist) "
        "that you may need later in this video. "
        "Set `pin=true` for *durable* facts that must always stay in "
        "the MEMORY brief (e.g. user constraints, character identities, "
        "scene type) — pinned facts are never evicted."
    )
    parameters = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": [
                "observation", "qa", "note", "tool", "audio", "fact"]},
            "content": {"type": "string"},
            "t": {"type": "integer", "description": "Frame second (defaults to current)."},
            "ref_frame": {"type": "integer"},
            "salience": {"type": "number", "description": "0.5-3.0 (default 1.0)."},
            "pin": {"type": "boolean",
                     "description": "Pin as a durable fact, never evicted."},
        },
        "required": ["content"],
    }

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        err = _require_memory(ctx)
        if err: return err
        kind = str(args.get("kind", "note"))
        content = str(args.get("content", ""))
        if not content:
            return ToolResult(self.name, False, error="empty content")
        t = int(args.get("t", ctx.current_round))
        ref = args.get("ref_frame")
        sal = float(args.get("salience", 1.0))
        pin = bool(args.get("pin", False))
        if pin:
            item = ctx.memory.pin_fact(content,
                                        kind=kind if kind != "note" else "fact",
                                        t=t)
            return ToolResult(self.name, True,
                              data={"pin_id": item["id"], "pinned": True,
                                    "t": item["t"]})
        ev = ctx.memory.write_event(t, kind, content,
                                    ref_frame=int(ref) if ref is not None else None,
                                    salience=sal)
        return ToolResult(self.name, True, data={"id": ev.eid, "t": ev.t})


class MemoryReadTool(Tool):
    name = "memory_read"
    description = (
        "Search memory for events matching a query (keyword overlap). "
        "Returns up to top_k events with their time, kind, text."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "description": "default 5"},
            "t_lo": {"type": "integer"},
            "t_hi": {"type": "integer"},
        },
        "required": ["query"],
    }

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        err = _require_memory(ctx)
        if err: return err
        q = str(args.get("query", ""))
        k = int(args.get("top_k", 5))
        rng = None
        if "t_lo" in args and "t_hi" in args:
            rng = (int(args["t_lo"]), int(args["t_hi"]))
        events = ctx.memory.search(q, top_k=k, t_range=rng)
        return ToolResult(self.name, True,
                          data={"events": [e.to_dict() for e in events]})


class MemoryListTool(Tool):
    name = "memory_list"
    description = (
        "List memory events filtered by kind / time range. Use this when "
        "you want a chronological view rather than keyword search."
    )
    parameters = {
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "t_lo": {"type": "integer"},
            "t_hi": {"type": "integer"},
            "limit": {"type": "integer", "description": "default 20"},
        },
        "required": [],
    }

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        err = _require_memory(ctx)
        if err: return err
        kind = args.get("kind")
        t_lo = args.get("t_lo")
        t_hi = args.get("t_hi")
        limit = int(args.get("limit", 20))
        evs = ctx.memory.list(
            kind=kind,
            t_lo=int(t_lo) if t_lo is not None else None,
            t_hi=int(t_hi) if t_hi is not None else None,
        )[:limit]
        return ToolResult(self.name, True,
                          data={"events": [e.to_dict() for e in evs]})


class MemoryForgetTool(Tool):
    name = "memory_forget"
    description = "Delete a memory event by id (for self-correction)."
    parameters = {
        "type": "object",
        "properties": {"id": {"type": "integer"}},
        "required": ["id"],
    }

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        err = _require_memory(ctx)
        if err: return err
        try:
            eid = int(args["id"])
        except (KeyError, TypeError, ValueError) as e:
            return ToolResult(self.name, False, error=f"bad id: {e}")
        ok = ctx.memory.forget(eid)
        return ToolResult(self.name, ok,
                          data={"removed": ok, "id": eid},
                          error=None if ok else "id not found")


class MemoryRetrieveTool(Tool):
    name = "memory_search"
    description = (
        "Cross-layer keyword retrieval over MEMORY (long_term + "
        "episodic + working). Returns up to top_k items per layer "
        "ranked by overlap × salience × layer_prior × change_boost / "
        "recency_decay. Prefer this over `memory_read` once the "
        "question has arrived — it surfaces durable pinned facts and "
        "aggregated episodes alongside per-frame events."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "description": "default 8"},
            "layers": {
                "type": "array",
                "items": {"type": "string",
                          "enum": ["long_term", "episodic", "working"]},
                "description": "Subset of layers to search; default all three.",
            },
            "t_lo": {"type": "integer"},
            "t_hi": {"type": "integer"},
        },
        "required": ["query"],
    }

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        err = _require_memory(ctx)
        if err: return err
        q = str(args.get("query", ""))
        if not q:
            return ToolResult(self.name, False, error="empty query")
        k = int(args.get("top_k", 8))
        layers = args.get("layers") or ["long_term", "episodic", "working"]
        layers = tuple(str(l) for l in layers)
        rng = None
        if "t_lo" in args and "t_hi" in args:
            rng = (int(args["t_lo"]), int(args["t_hi"]))
        result = ctx.memory.retrieve(
            q, top_k=k, t_range=rng,
            current_t=int(ctx.current_round),
            layers=layers,
        )
        # ----- 改动 1: 命中重复检测 + 死循环阻断 -----
        # 提取本次返回的 top_id 集合（跨层）
        cur_ids = set()
        if isinstance(result, dict):
            for layer_name in ("long_term", "episodic", "working"):
                for item in (result.get(layer_name) or []):
                    if isinstance(item, dict) and "id" in item:
                        cur_ids.add((layer_name, item["id"]))
        history = ctx.memsearch_history
        # 与上一次查询比较：top-id 重叠 >= 80% 视为"等价查询"
        is_repeat = False
        prev_query = None
        if history and cur_ids:
            last = history[-1]
            prev_ids = last.get("top_ids") or set()
            if prev_ids:
                inter = len(cur_ids & prev_ids)
                ratio = inter / max(1, len(cur_ids | prev_ids))
                if ratio >= 0.8:
                    is_repeat = True
                    prev_query = last.get("query")
        # 同物理轮 memory_search 累计 >= 3 → 拒收
        if len(history) >= 3:
            return ToolResult(
                self.name, False,
                error=(
                    "memory_search exhausted on this physical round "
                    f"({len(history)} prior calls returned the same evidence). "
                    "Stop searching — switch tool family: "
                    "(a) `replay(t=<frame>)` to revisit a specific moment "
                    "you remember from MEMORY; "
                    "(b) commit `</Response> <answer>` with current evidence "
                    "— more searches will NOT yield new information."
                ),
            )
        history.append({"query": q, "top_ids": cur_ids})
        if is_repeat:
            # 调用成功但附加 guidance，让模型停止重复搜索
            if isinstance(result, dict):
                result = dict(result)  # shallow copy 防意外突变
                result["is_repeat_of_query"] = prev_query
                result["guidance"] = (
                    "this query returns the SAME evidence as your previous "
                    f"memory_search ('{prev_query}'). Re-querying will not "
                    "yield new information. To gather DIFFERENT evidence: "
                    "use `replay(t=<frame_mentioned_above>)` to revisit a "
                    "specific frame, OR commit `</Response>` now."
                )
        return ToolResult(self.name, True, data=result)


class MemoryUnpinTool(Tool):
    name = "memory_unpin"
    description = (
        "Remove a previously pinned fact by its pin_id. Use when a "
        "pinned constraint is no longer valid (scene changed, "
        "hypothesis disproved)."
    )
    parameters = {
        "type": "object",
        "properties": {"pin_id": {"type": "integer"}},
        "required": ["pin_id"],
    }

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        err = _require_memory(ctx)
        if err: return err
        try:
            pid = int(args["pin_id"])
        except (KeyError, TypeError, ValueError) as e:
            return ToolResult(self.name, False, error=f"bad pin_id: {e}")
        ok = ctx.memory.unpin_fact(pid)
        return ToolResult(self.name, ok,
                          data={"removed": ok, "pin_id": pid},
                          error=None if ok else "pin_id not found")
