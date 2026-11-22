"""Pure replay of one attempt's engine journal into blocks, requests, work, decisions, and episodes.

Replay applies recorded effects and checks every prior value it can. It never initializes a later
state from the snapshot it is checking: only the initial snapshot seeds the pool, and every later
snapshot and state digest is compared field by field. The first mismatch makes the run invalid.

Completed KV comes only from the epoch ledger: a Work item advances its request's completed
coverage when a matching valid completion arrives for the same epoch. Stale and orphaned
completions are executed-discarded work. The engine's computed-minus-in-flight value is kept only
as a cross-check count.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from array import array
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NamedTuple, TypedDict, get_type_hints

from e02 import events as ev

FINISHED_PREFIX = "FINISHED_"
PRODUCER_ERROR = ev.PRODUCER_ERROR
EpisodeStatus = Literal["waiting", "resumed", "regained", "preempted_again", "finished", "aborted", "unresolved"]
WorkStatus = Literal["scheduled", "completed_valid", "completed_discarded", "unacknowledged"]
Sufficiency = Literal["yes", "no", "unknown"]
RunStatus = Literal["complete", "incomplete", "invalid"]
StepKind = Literal["decode", "mixed", "prefill"]
FinalInterval = Literal["completed", "unacknowledged", "none"]
SnapshotReason = Literal["initial", "periodic", "decision", "terminal"]
BATCH_TIME_BASIS = ("host schedule-time proxy: each step lasts until the next schedule call, the final step until "
                    "its completion; device time unavailable")


class ReplayError(Exception):
    def __init__(self, seq: int, message: str):
        super().__init__(f"seq {seq}: {message}")
        self.seq = seq
        self.message = message


class EvidenceError(Exception):
    pass


class SnapshotBlock(NamedTuple):
    block: int
    ref: int
    keys: int
    keys_digest: str


class SnapshotRequest(NamedTuple):
    r: int
    status: str
    table: list[int]
    computed: int
    in_flight: int
    stale: int
    num_tokens: int
    num_output: int
    preemptions: int
    releasable: bool


class SnapshotRecord(TypedDict):
    index: int
    reason: SnapshotReason
    call: int
    seq: int
    blocks: list[SnapshotBlock]
    queue: list[int]
    free: int
    hash_keys: int
    requests: list[SnapshotRequest]
    running: list[int]
    waiting: list[int]
    kv_holding_waiting: int
    victim: int | None


@dataclass(frozen=True)
class BlockIdentity:
    block: int
    generation: int


@dataclass
class BlockState:
    ref: int = 0
    generation: int = 0
    keys: set = field(default_factory=set)
    last_hash_del: int = -1


@dataclass
class Work:
    out: int
    r: int
    epoch: int
    start: int
    end: int
    blocks: tuple
    t_scheduled: int
    status: WorkStatus = "scheduled"
    reprefill: int = 0


@dataclass
class RequestState:
    r: int
    engine_id: str
    prompt_len: int
    prompt_sha: str
    max_tokens: int
    t_arrival: int
    status: str = "WAITING"
    table: list = field(default_factory=list)
    computed: int = 0
    in_flight: int = 0
    stale: int = 0
    num_tokens: int = 0
    num_output: int = 0
    preemptions: int = 0
    completed_end: int = 0
    greatest: int = 0
    pending: dict = field(default_factory=dict)
    first_admit_t: int | None = None
    last_admit_t: int | None = None
    reprefill_tokens: int = 0
    discarded_tokens: int = 0
    episodes: list = field(default_factory=list)
    finish_t: int | None = None
    final_n_out: int | None = None
    pending_lookup: tuple | None = None


class HitBlock(NamedTuple):
    block: int
    generation: int


class OriginalBlock(NamedTuple):
    position: int
    block: int
    generation: int
    ref_after_release: int


@dataclass
class LookupAttempt:
    call: int
    t: int
    hit_tokens: int
    admitted: bool = False


@dataclass
class Recovery:
    lookup_call: int
    t_resume: int
    hit_tokens: int
    H: int
    original_freed: int
    original_shared: int
    equivalent: int
    unknown: int
    missing: int
    intervening_alloc_blocks: int
    freed_blocks_reallocated: int
    hit_blocks: list[HitBlock]


@dataclass
class PreemptionEpisode:
    id: str
    r: int
    ordinal: int
    decision: int
    snapshot: int
    call: int
    t: int
    C: int
    raw: int
    pending_tokens: int
    originals: list[OriginalBlock]
    reclaimed_blocks: int
    lookups: list[LookupAttempt] = field(default_factory=list)
    recovery: Recovery | None = None
    regained_t: int | None = None
    reprefill_tokens: int = 0
    status: EpisodeStatus = "waiting"
    alloc_pops_at_release: int = 0
    release_seq: int = -1

    @property
    def fraction(self) -> float | None:
        if self.C == 0 or self.recovery is None:
            return None
        return self.recovery.H / self.C


@dataclass
class Candidate:
    decision: int
    r: int
    chosen: bool
    eligible: bool
    completed: int
    raw: int
    pending: int
    rank: int | None
    least_tied: bool
    total_blocks: int
    shared_blocks: int
    reclaimable_blocks: int
    reserved_blocks: int
    shared_prefix_tokens: int
    sufficiency: Sufficiency
    sufficiency_reason: str
    running_position: int | None
    first_admit_t: int | None
    last_admit_t: int | None
    preemptions_before: int


@dataclass
class DecisionRecord:
    d: int
    call: int
    t: int
    snapshot: int
    victim: int
    requester: int
    alloc: int
    needed: int | None
    free: int
    running_before: list
    stock_rule_holds: bool
    candidates: list
    reclaimed_blocks: int | None = None
    episode: str | None = None
    group: int = 0


@dataclass
class SchedStep:
    """One forward pass: a schedule output with at least one request in it."""
    out: int
    call: int
    t: int
    batch: int
    tokens: int
    prefill_entries: int
    kind: StepKind
    t_done: int | None = None
    t_next: int | None = None


@dataclass
class Replay:
    run_id: str
    config: dict
    num_blocks: int
    null_block: int
    block_size: int
    requests: dict
    decisions: list
    episodes: dict
    steps: list
    works_unacknowledged: int
    hint_divergences: int
    snapshots_checked: int
    digests_checked: int
    alloc_failures: int
    end: ev.RunEnd | None
    blocks: dict
    snapshots: dict[int, tuple[str, SnapshotRecord]]


class Reducer:
    def __init__(self, snapshots: dict[int, tuple[str, dict]]):
        self.snapshots = snapshots
        self.blocks: dict[int, BlockState] = {}
        self.queue: list[int] = []
        self.key_blocks: dict[str, set] = defaultdict(set)
        self.requests: dict[int, RequestState] = {}
        self.running: list[int] = []
        self.waiting: list[int] = []
        self.decisions: list[DecisionRecord] = []
        self.episodes: dict[str, PreemptionEpisode] = {}
        self.steps: list[SchedStep] = []
        self.step_by_out: dict[int, SchedStep] = {}
        self.allocs: dict[int, ev.Alloc] = {}
        self.alloc_pops = 0
        self.open_decision: tuple[DecisionRecord, list] | None = None
        self.hint_divergences = 0
        self.snapshots_checked = 0
        self.digests_checked = 0
        self.alloc_failures = 0
        self.started = False
        self.start: ev.RunStart | None = None
        self.last_failed_group = None
        self.group = 0
        self.seq = -1
        self.t = 0
        self.call = 0

    def fail(self, msg: str):
        raise ReplayError(self.seq, msg)

    def req(self, r: int) -> RequestState:
        if r not in self.requests:
            self.fail(f"unknown request {r}")
        return self.requests[r]

    def apply(self, e: ev.Event) -> None:
        self.seq, self.t, self.call = e.seq, e.t_ns, e.call
        p = e.payload
        if not self.started and not isinstance(p, ev.RunStart):
            self.fail(f"{p.KIND} before RunStart")
        getattr(self, "on_" + p.KIND)(p)

    def on_start(self, p: ev.RunStart):
        if self.started:
            self.fail("second RunStart")
        self.started = True
        self.start = p
        self.null = p.null_block
        self.bs = p.block_size

    def on_snap(self, p: ev.SnapshotTaken):
        if p.index not in self.snapshots:
            self.fail(f"snapshot {p.index} missing from snapshots.jsonl")
        line, rec = self.snapshots[p.index]
        if hashlib.sha256(line.encode()).hexdigest() != p.digest:
            self.fail(f"snapshot {p.index} digest mismatch")
        if rec["reason"] != p.reason or rec["seq"] != self.seq:
            self.fail(f"snapshot {p.index} header does not match its event")
        if rec["reason"] == "initial":
            self.seed(rec)
        else:
            self.compare(rec)
        self.snapshots_checked += 1

    def seed(self, rec: dict):
        if rec["blocks"] or rec["requests"] or rec["hash_keys"]:
            self.fail("initial snapshot is not an empty cache")
        for b in range(self.start.num_blocks):
            self.blocks[b] = BlockState()
        self.queue = list(rec["queue"])
        if sorted(self.queue) != [b for b in range(self.start.num_blocks) if b != self.null]:
            self.fail("initial free queue does not hold every usable block once")

    def compare(self, rec: dict):
        mine = self.block_rows()
        if mine != rec["blocks"]:
            diff = next((m, s) for m, s in zip(mine + [None] * len(rec["blocks"]), rec["blocks"] + [None] * len(mine)) if m != s)
            self.fail(f"snapshot {rec['index']} blocks differ, first (replay, direct) = {diff}")
        if self.queue != rec["queue"]:
            self.fail(f"snapshot {rec['index']} free queue order differs")
        if len(self.queue) != rec["free"]:
            self.fail(f"snapshot {rec['index']} free count differs")
        if self.map_size() != rec["hash_keys"]:
            self.fail(f"snapshot {rec['index']} hash key count differs")
        live = {r: q for r, q in self.requests.items() if not q.status.startswith(FINISHED_PREFIX)}
        direct = {row[0]: row for row in rec["requests"]}
        if set(direct) != set(live):
            self.fail(f"snapshot {rec['index']} live requests differ: replay {sorted(live)} direct {sorted(direct)}")
        for r, row in direct.items():
            q = live[r]
            _, status, table, computed, in_flight, stale, num_tokens, num_output, preemptions, _ = row
            mine = [q.status, q.table, q.computed, q.in_flight, q.stale, q.num_tokens, q.num_output, q.preemptions]
            theirs = [status, table, computed, in_flight, stale, num_tokens, num_output, preemptions]
            if mine != theirs:
                self.fail(f"snapshot {rec['index']} request {r} differs: replay {mine} direct {theirs}")
        if rec["reason"] != "decision" and (self.running != rec["running"] or self.waiting != rec["waiting"]):
            self.fail(f"snapshot {rec['index']} running or waiting order differs")

    def block_rows(self) -> list:
        rows = []
        for b, s in self.blocks.items():
            if b == self.null:
                continue
            if s.ref or s.keys:
                rows.append([b, s.ref, len(s.keys), keys_digest(s.keys) if s.keys else ""])
        return rows

    def map_size(self) -> int:
        return sum(1 for v in self.key_blocks.values() if v)

    def on_digest(self, p: ev.StateDigest):
        refs = array("i", (s.ref for b, s in self.blocks.items() if b != self.null))
        head = self.queue[0] if self.queue else -1
        tail = self.queue[-1] if self.queue else -1
        info = array("i", [len(self.queue), head, tail, self.map_size()])
        if hashlib.blake2b(refs.tobytes() + info.tobytes(), digest_size=8).hexdigest() != p.value:
            self.fail("state digest differs from replay")
        self.digests_checked += 1

    def on_fault(self, p: ev.Fault):
        self.fail(f"observer fault at {p.where}: {p.error}")

    def on_foot(self, p):
        pass

    def block(self, b: int) -> BlockState:
        if b not in self.blocks:
            self.fail(f"unknown block {b}")
        return self.blocks[b]

    def on_qpop(self, p: ev.QPop):
        k = len(p.blocks)
        if self.queue[:k] != p.blocks:
            self.fail(f"allocation popped {p.blocks}, replayed queue head is {self.queue[:k]}")
        del self.queue[:k]

    def on_qrm(self, p: ev.QRemove):
        try:
            self.queue.remove(p.block)
        except ValueError:
            self.fail(f"removed block {p.block} is not in the free queue")

    def on_qpre(self, p: ev.QPrepend):
        self.queue[:0] = p.blocks

    def on_qapp(self, p: ev.QAppend):
        self.queue.extend(p.blocks)

    def on_new(self, p: ev.PoolAlloc):
        for b in p.blocks:
            s = self.block(b)
            if s.ref != 0 or s.keys:
                self.fail(f"allocated block {b} has ref {s.ref} and {len(s.keys)} hashes")
            s.ref = 1
            s.generation += 1
        self.alloc_pops += len(p.blocks)

    def on_touch(self, p: ev.PoolTouch):
        if len(p.blocks) != len(p.refs_before):
            self.fail("touch lists differ in length")
        for b, before in zip(p.blocks, p.refs_before):
            s = self.block(b)
            if s.ref != before:
                self.fail(f"touch of block {b}: replay ref {s.ref}, engine {before}")
            s.ref += 1

    def on_free(self, p: ev.PoolFree):
        if len(p.blocks) != len(p.refs_before):
            self.fail("free lists differ in length")
        for b, before in zip(p.blocks, p.refs_before):
            s = self.block(b)
            if s.ref != before:
                self.fail(f"free of block {b}: replay ref {s.ref}, engine {before}")
            s.ref -= 1

    def on_hadd(self, p: ev.HashAdd):
        s = self.block(p.block)
        if p.key in s.keys or (p.primary and s.keys):
            self.fail(f"hash add on block {p.block} contradicts its replayed keys")
        s.keys.add(p.key)
        self.key_blocks[p.key].add(p.block)

    def on_hdel(self, p: ev.HashDel):
        s = self.block(p.block)
        if set(p.keys) != s.keys:
            self.fail(f"hash removal on block {p.block} names keys the replay does not hold")
        for k in p.keys:
            self.key_blocks[k].discard(p.block)
        s.keys = set()
        s.last_hash_del = self.seq

    def on_creset(self, p):
        if any(s.ref for b, s in self.blocks.items() if b != self.null):
            self.fail("cache reset with referenced blocks")
        for s in self.blocks.values():
            s.keys = set()
        self.key_blocks.clear()

    def on_add(self, p: ev.ReqAdd):
        if p.r in self.requests:
            self.fail(f"request {p.r} added twice")
        self.requests[p.r] = RequestState(p.r, p.engine_id, p.prompt_len, p.prompt_sha, p.max_tokens, self.t,
                                          num_tokens=p.prompt_len)

    def on_queues(self, p: ev.Queues):
        self.running, self.waiting = list(p.running), list(p.waiting)

    def on_lookup(self, p: ev.Lookup):
        q = self.req(p.r)
        gens = [self.block(b).generation for b in p.hit_blocks]
        q.pending_lookup = (self.call, self.t, p.hit_tokens, list(p.hit_blocks), gens, self.alloc_pops)
        if q.status == "PREEMPTED":
            self.episodes[q.episodes[-1]].lookups.append(LookupAttempt(self.call, self.t, p.hit_tokens))

    def on_alloc(self, p: ev.Alloc):
        q = self.req(p.r)
        self.allocs[p.a] = p
        if not p.ok:
            self.alloc_failures += 1
            if p.appended:
                self.fail(f"failed allocation {p.a} changed the block table")
            return
        if q.status in ("WAITING", "PREEMPTED") and p.n_hit != (q.pending_lookup[2] if q.pending_lookup else 0):
            self.fail(f"admission of request {p.r} adopts {p.n_hit} hit tokens, lookup found "
                      f"{q.pending_lookup[2] if q.pending_lookup else None}")
        for b in p.appended:
            self.block(b)
        q.table.extend(p.appended)
        if q.status == "PREEMPTED":
            self.resume(q)

    def resume(self, q: RequestState):
        ep = self.episodes[q.episodes[-1]]
        if q.pending_lookup is None:
            self.fail(f"request {q.r} resumed without a lookup")
        call, t, hit_tokens, hit_blocks, gens, pops = q.pending_lookup
        if ep.lookups:
            ep.lookups[-1].admitted = True
        bs = self.bs
        H = min(hit_tokens, ep.C)
        if len(hit_blocks) * bs < H:
            self.fail(f"request {q.r} lookup found {hit_tokens} tokens in {len(hit_blocks)} blocks")
        parts = {"original_freed": 0, "original_shared": 0, "equivalent": 0, "unknown": 0}
        for i in range((H + bs - 1) // bs):
            tokens = min(bs, H - i * bs)
            hit_block, hit_gen = hit_blocks[i], gens[i]
            if i >= len(ep.originals):
                parts["unknown"] += tokens
                continue
            orig = ep.originals[i]
            if (hit_block, hit_gen) != (orig.block, orig.generation):
                parts["equivalent"] += tokens
            elif self.blocks[hit_block].last_hash_del > ep.release_seq:
                parts["unknown"] += tokens
            elif orig.ref_after_release == 0:
                parts["original_freed"] += tokens
            else:
                parts["original_shared"] += tokens
        reallocated = sum(1 for o in ep.originals
                          if o.ref_after_release == 0 and self.blocks[o.block].generation != o.generation)
        ep.recovery = Recovery(lookup_call=call, t_resume=self.t, hit_tokens=hit_tokens, H=H, missing=ep.C - H,
                               intervening_alloc_blocks=pops - ep.alloc_pops_at_release,
                               freed_blocks_reallocated=reallocated,
                               hit_blocks=[HitBlock(b, g) for b, g in zip(hit_blocks, gens)], **parts)
        ep.status = "resumed"
        if ep.C <= H:
            ep.regained_t = self.t
            ep.status = "regained"

    def on_detach(self, p: ev.Detach):
        q = self.req(p.r)
        if q.table != p.blocks:
            self.fail(f"release of request {p.r}: replay table {q.table}, engine {p.blocks}")
        q.table = []

    def on_sched(self, p: ev.Scheduled):
        admitted = set(p.new) | set(p.resumed)
        prefill = 0
        for r, n, start, epoch in p.entries:
            q = self.req(r)
            if r in admitted:
                expected = "WAITING" if r in p.new else "PREEMPTED"
                if q.status != expected:
                    self.fail(f"request {r} admitted from {q.status}, expected {expected}")
                q.status = "RUNNING"
                q.computed = start
                q.completed_end = start
                q.greatest = max(q.greatest, start)
                q.first_admit_t = q.first_admit_t if q.first_admit_t is not None else self.t
                q.last_admit_t = self.t
            elif q.status != "RUNNING" or q.computed != start:
                self.fail(f"request {r} scheduled at {start} from {q.status}, replay computed {q.computed}")
            if epoch != q.preemptions:
                self.fail(f"request {r} scheduled in epoch {epoch}, replay epoch {q.preemptions}")
            blocks = tuple(BlockIdentity(b, self.block(b).generation)
                           for b in q.table[start // self.bs:(start + n - 1) // self.bs + 1])
            if len(blocks) != (start + n - 1) // self.bs - start // self.bs + 1:
                self.fail(f"request {r} scheduled past its block table")
            q.pending[p.out] = Work(p.out, r, epoch, start, start + n, blocks, self.t)
            q.computed += n
            q.in_flight += n
            prefill += n > 1 or start < q.prompt_len
        if self.steps and self.steps[-1].t_next is None:
            self.steps[-1].t_next = self.t
        if p.entries:
            step = SchedStep(p.out, self.call, self.t, len(p.entries), p.total, prefill,
                             step_kind(len(p.entries), prefill))
            self.steps.append(step)
            self.step_by_out[p.out] = step

    def on_done(self, p: ev.Completed):
        step = self.step_by_out.get(p.out)
        if step is not None:
            if step.t_done is not None:
                self.fail(f"output {p.out} completed twice")
            step.t_done = self.t
        for r, n, kind in p.entries:
            q = self.req(r)
            w = q.pending.pop(p.out, None)
            if w is None or w.end - w.start != n:
                self.fail(f"completion of output {p.out} for request {r} has no matching scheduled work")
            same_epoch = w.epoch == q.preemptions
            if kind == "gone":
                if not q.status.startswith(FINISHED_PREFIX):
                    self.fail(f"engine reports request {r} gone, replay status {q.status}")
                w.status = "completed_discarded"
                q.discarded_tokens += n
                continue
            q.in_flight -= n
            if kind == "stale":
                if same_epoch or q.stale < n:
                    self.fail(f"stale completion for request {r} in its current epoch or beyond its stale share")
                q.stale -= n
                w.status = "completed_discarded"
                q.discarded_tokens += n
            elif kind == "valid":
                if not same_epoch or q.status != "RUNNING":
                    self.fail(f"valid completion for request {r} from epoch {w.epoch}, replay epoch {q.preemptions}")
                if w.start != q.completed_end:
                    self.fail(f"request {r} completed [{w.start},{w.end}) but coverage ends at {q.completed_end}")
                w.status = "completed_valid"
                w.reprefill = max(0, min(w.end, q.greatest) - w.start)
                q.reprefill_tokens += w.reprefill
                q.completed_end = w.end
                q.greatest = max(q.greatest, w.end)
                if q.episodes:
                    ep = self.episodes[q.episodes[-1]]
                    ep.reprefill_tokens += w.reprefill
                    if ep.regained_t is None and ep.recovery is not None and q.completed_end >= ep.C:
                        ep.regained_t = self.t
                        ep.status = "regained"

    def on_delivered(self, p: ev.Delivered):
        for r, delivered, hint in p.entries:
            q = self.req(r)
            q.num_output += delivered
            q.num_tokens += delivered
            if not q.status.startswith(FINISHED_PREFIX) and q.status == "RUNNING" and hint != q.completed_end:
                self.hint_divergences += 1

    def on_decide(self, p: ev.Decision):
        if p.snapshot not in self.snapshots:
            self.fail(f"decision {p.d} names missing snapshot {p.snapshot}")
        _, rec = self.snapshots[p.snapshot]
        if rec["victim"] != p.victim:
            self.fail(f"decision {p.d} snapshot names victim {rec['victim']}, event names {p.victim}")
        alloc = self.allocs.get(p.a)
        if alloc is None or alloc.ok:
            self.fail(f"decision {p.d} has no failed allocation")
        if self.last_failed_group != (alloc.r, self.call):
            self.group += 1
            self.last_failed_group = (alloc.r, self.call)
        needed = alloc.needed[-1] if alloc.needed else None
        refs = {row[0]: row[1] for row in rec["blocks"]}
        rows = {row[0]: row for row in rec["requests"]}
        cands = [r for r in rec["running"]] + [p.victim]
        if p.victim in rec["running"]:
            self.fail(f"decision {p.d}: victim still in the running list")
        if not set(cands) <= set(rows):
            self.fail(f"decision {p.d}: snapshot lacks a candidate's request row")
        completed = {r: self.req(r).completed_end for r in cands}
        eligible = {r: rows[r][9] for r in cands}
        least = min((completed[r] for r in cands if eligible[r]), default=None)
        out = []
        for r in cands:
            q = self.req(r)
            table = rows[r][2]
            real = [b for b in table if b != self.null]
            shared = sum(1 for b in real if refs.get(b, 0) > 1)
            reclaimable = sum(1 for b in real if refs.get(b, 0) == 1) if eligible[r] else 0
            prefix_blocks = 0
            for b in table:
                if b == self.null or refs.get(b, 0) <= 1:
                    break
                prefix_blocks += 1
            suff, why = sufficiency(r, alloc, needed, rec["free"], reclaimable, eligible[r], self.start.config)
            rank = None
            if eligible[r]:
                rank = 1 + sum(1 for x in cands if eligible[x] and completed[x] < completed[r])
            out.append(Candidate(
                decision=p.d, r=r, chosen=r == p.victim, eligible=eligible[r], completed=completed[r],
                raw=rows[r][3], pending=rows[r][4], rank=rank,
                least_tied=eligible[r] and completed[r] == least, total_blocks=len(set(real)),
                shared_blocks=shared, reclaimable_blocks=reclaimable, reserved_blocks=len(table) - len(real),
                shared_prefix_tokens=min(completed[r], prefix_blocks * self.bs), sufficiency=suff,
                sufficiency_reason=why,
                running_position=p.running.index(r) if p.running and r in p.running else None,
                first_admit_t=q.first_admit_t, last_admit_t=q.last_admit_t, preemptions_before=q.preemptions))
        dec = DecisionRecord(d=p.d, call=self.call, t=self.t, snapshot=p.snapshot, victim=p.victim,
                             requester=p.requester, alloc=p.a, needed=needed, free=rec["free"],
                             running_before=list(p.running or []),
                             stock_rule_holds=bool(p.running) and p.running[-1] == p.victim,
                             candidates=out, group=self.group)
        victim = self.req(p.victim)
        originals = [(i, b, self.block(b).generation) for i, b in enumerate(victim.table)]
        self.decisions.append(dec)
        self.open_decision = (dec, originals)

    def on_preempt(self, p: ev.Preempted):
        if self.open_decision is None or self.open_decision[0].d != p.d:
            self.fail(f"preemption {p.d} without its decision")
        dec, originals = self.open_decision
        self.open_decision = None
        q = self.req(p.r)
        if (q.computed, q.in_flight) != (p.raw, p.in_flight):
            self.fail(f"preempted request {p.r}: replay raw {q.computed} in-flight {q.in_flight}, engine {p.raw} {p.in_flight}")
        if q.table:
            self.fail(f"preempted request {p.r} still holds blocks after release")
        if q.preemptions + 1 != p.preemptions or p.stale != q.in_flight:
            self.fail(f"preempted request {p.r}: count or stale share differs")
        if len(self.queue) != p.free_after:
            self.fail(f"free count after preemption {p.d} differs from replay")
        if q.episodes:
            prev = self.episodes[q.episodes[-1]]
            if prev.status in ("resumed", "regained", "waiting"):
                prev.status = "preempted_again"
        C = q.completed_end
        ep = PreemptionEpisode(
            id=f"{p.r}.{p.preemptions}", r=p.r, ordinal=p.preemptions, decision=p.d, snapshot=dec.snapshot,
            call=self.call, t=self.t, C=C, raw=p.raw, pending_tokens=sum(w.end - w.start for w in q.pending.values()),
            originals=[OriginalBlock(i, b, g, self.blocks[b].ref) for i, b, g in originals],
            reclaimed_blocks=p.free_after - p.free_before, alloc_pops_at_release=self.alloc_pops,
            release_seq=self.seq)
        self.episodes[ep.id] = ep
        q.episodes.append(ep.id)
        dec.reclaimed_blocks = p.free_after - p.free_before
        dec.episode = ep.id
        q.status = "PREEMPTED"
        q.stale = q.in_flight
        q.computed = 0
        q.completed_end = 0
        q.preemptions = p.preemptions
        q.pending_lookup = None

    def on_finish(self, p: ev.Finished):
        q = self.req(p.r)
        if q.status.startswith(FINISHED_PREFIX):
            self.fail(f"request {p.r} finished twice")
        q.status = p.status
        q.finish_t = self.t
        q.final_n_out = p.n_out
        if q.episodes:
            ep = self.episodes[q.episodes[-1]]
            if ep.status in ("waiting", "resumed", "regained"):
                ep.status = "aborted" if p.status == "FINISHED_ABORTED" else "finished"

    def finish(self, end: ev.RunEnd | None) -> Replay:
        unack = 0
        for q in self.requests.values():
            for w in q.pending.values():
                w.status = "unacknowledged"
                unack += 1
            if q.final_n_out is not None and q.final_n_out != q.num_output:
                self.fail(f"request {q.r} finished with {q.final_n_out} outputs, replay delivered {q.num_output}")
        for ep in self.episodes.values():
            if ep.status in ("waiting", "resumed"):
                ep.status = "unresolved"
        s = self.start
        return Replay(s.run_id, s.config, s.num_blocks, s.null_block, s.block_size, self.requests, self.decisions,
                      self.episodes, self.steps, unack, self.hint_divergences, self.snapshots_checked,
                      self.digests_checked, self.alloc_failures, end, self.blocks, self.snapshots)


def step_kind(batch: int, prefill_entries: int) -> StepKind:
    if prefill_entries == 0:
        return "decode"
    return "prefill" if prefill_entries == batch else "mixed"


def keys_digest(keys) -> str:
    return hashlib.blake2b("|".join(sorted(keys)).encode(), digest_size=8).hexdigest()


def sufficiency(r, alloc: ev.Alloc, needed, free, reclaimable, eligible, config) -> tuple[Sufficiency, str]:
    """Pure check of whether releasing r alone lets the failed running-path allocation succeed.
    Covers only the constraints of the pinned single full-attention group configuration."""
    if r == alloc.r:
        return "unknown", "requester"
    if not eligible:
        return "unknown", "release deferred"
    if needed is None or len(alloc.needed) != 1:
        return "unknown", "allocation count not captured"
    if alloc.full_fit or alloc.reserved or config.get("watermark_blocks") or config.get("num_lookahead_tokens"):
        return "unknown", "admission, reserve, watermark, or lookahead constraint"
    if config.get("kv_groups") != ["FullAttentionManager"]:
        return "unknown", "non-full-attention group"
    return ("yes" if free + reclaimable >= needed else "no"), f"free {free} + reclaimable {reclaimable} vs needed {needed}"


@dataclass(frozen=True)
class Invalid:
    reason: str
    seq: int | None


def read_snapshots(path: Path) -> dict[int, tuple[str, SnapshotRecord]]:
    """Every snapshot line, keyed by index. A missing file is empty; any malformed, partial,
    mistyped, duplicate or out-of-order line raises EvidenceError."""
    if not path.exists():
        return {}
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise EvidenceError("snapshots.jsonl ends with a partial line")
    fields = get_type_hints(SnapshotRecord)
    out = {}
    for i, raw in enumerate(data.split(b"\n")[:-1]):
        try:
            line = raw.decode()
            rec = json.loads(line)
            if type(rec) is not dict or set(rec) != set(fields):
                raise ev.ShapeError(f"expected an object with keys {sorted(fields)}")
            for name, tp in fields.items():
                try:
                    ev.parse_value(tp, rec[name])
                except ev.ShapeError as e:
                    raise ev.ShapeError(f"{name}: {e}") from None
        except ValueError as e:
            raise EvidenceError(f"snapshots.jsonl line {i}: {e}") from None
        if rec["index"] != i:
            raise EvidenceError(f"snapshots.jsonl line {i}: index {rec['index']} differs from its line number")
        out[i] = (line, rec)
    return out


def producer_error(directory: Path) -> str | None:
    path = directory / PRODUCER_ERROR
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text())
        return f"{rec['where']}: {rec['error']}"
    except (OSError, ValueError, TypeError, KeyError) as e:
        return f"unreadable {PRODUCER_ERROR}: {e!r}"


def replay_attempt(directory: Path) -> tuple[ev.Journal, Replay | Invalid]:
    journal = ev.read_journal(directory / "events.jsonl")
    producer = producer_error(directory)
    if producer is not None:
        return journal, Invalid(f"producer error: {producer}", None)
    if journal.status == "corrupt":
        return journal, Invalid(f"journal corrupt: {journal.reason}", journal.bad_line)
    try:
        snapshots = read_snapshots(directory / "snapshots.jsonl")
    except (EvidenceError, OSError) as e:
        return journal, Invalid(f"snapshots invalid: {e}", None)
    red = Reducer(snapshots)
    try:
        for e in journal.events:
            red.apply(e)
        if not red.started:
            return journal, Invalid("journal has no RunStart", 0)
        if journal.status == "sealed" and journal.end.snapshots != len(red.snapshots):
            raise ReplayError(journal.end.n, "snapshot count differs from the seal")
        return journal, red.finish(journal.end)
    except ReplayError as e:
        return journal, Invalid(e.message, e.seq)
    except (LookupError, ValueError, TypeError, AttributeError) as e:
        return journal, Invalid(f"evidence breaks replay: {e!r}", red.seq)


def _intervals_ns(steps: list[SchedStep]) -> list[int | None]:
    out: list[int | None] = [(a.t_next if a.t_next is not None else b.t) - a.t for a, b in zip(steps, steps[1:])]
    if steps:
        out.append(None if _final_end(steps[-1]) is None else _final_end(steps[-1]) - steps[-1].t)
    return out


def _final_end(step: SchedStep) -> int | None:
    return step.t_next if step.t_next is not None else step.t_done


@dataclass(frozen=True)
class BatchBin:
    kind: StepKind
    batch: int
    steps: int
    seconds: float


def batch_distribution(steps: list[SchedStep]) -> tuple[tuple[BatchBin, ...], FinalInterval]:
    """Steps and seconds per (kind, batch size) under BATCH_TIME_BASIS. A final step that was never
    acknowledged is counted with no time."""
    acc: dict[tuple[StepKind, int], tuple[int, int]] = {}
    for s, ns in zip(steps, _intervals_ns(steps)):
        n, total = acc.get((s.kind, s.batch), (0, 0))
        acc[(s.kind, s.batch)] = (n + 1, total + (ns or 0))
    bins = tuple(BatchBin(kind, batch, n, total / 1e9) for (kind, batch), (n, total) in sorted(acc.items()))
    final: FinalInterval = ("none" if not steps else "completed" if _final_end(steps[-1]) is not None
                            else "unacknowledged")
    return bins, final


def time_weighted_batch(bins) -> float | None:
    seconds = sum(b.seconds for b in bins)
    return sum(b.batch * b.seconds for b in bins) / seconds if seconds else None


CANDIDATE_FIELDS = ("completed", "total_blocks", "shared_blocks", "reclaimable_blocks", "shared_prefix_tokens")


@dataclass(frozen=True)
class Distributions:
    batch: tuple[BatchBin, ...]
    batch_basis: str
    batch_final_interval: FinalInterval
    candidate: dict[str, tuple[int, ...]]
    recovery_fraction: tuple[float, ...]
    episode_reprefill_tokens: tuple[int, ...]


def distributions(rep: Replay) -> Distributions:
    """Per-run distributions: one value per candidate row in decision order, and per episode in
    (request, ordinal) order."""
    bins, final = batch_distribution(rep.steps)
    rows = [c for d in rep.decisions for c in d.candidates]
    eps = sorted(rep.episodes.values(), key=lambda e: (e.r, e.ordinal))
    return Distributions(
        batch=bins, batch_basis=BATCH_TIME_BASIS, batch_final_interval=final,
        candidate={f: tuple(getattr(c, f) for c in rows) for f in CANDIDATE_FIELDS},
        recovery_fraction=tuple(e.fraction for e in eps if e.fraction is not None),
        episode_reprefill_tokens=tuple(e.reprefill_tokens for e in eps))


STALL_NS = 5_000_000_000
ClientStatus = Literal["ok", "http_error", "error", "timeout", "cancelled"]


@dataclass
class ClientRecord:
    """One streamed request as the client saw it. Times are time.monotonic_ns(), which on macOS
    is the same system-wide clock the engine journal uses."""
    client_id: str
    item: int
    order: int
    phase: str
    t_submit_ns: int
    t_first_ns: int | None
    t_last_ns: int | None
    t_end_ns: int
    status: ClientStatus
    http_status: int | None
    error: str | None
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    chunks: int
    content_chunks: int
    reasoning_chunks: int
    max_gap_ns: int | None
    response_id: str | None


def read_client(path: Path) -> tuple[list[ClientRecord], str | None]:
    if not path.exists():
        return [], "client.jsonl missing"
    try:
        lines = path.read_text().splitlines()
    except (OSError, ValueError) as e:
        return [], f"client.jsonl unreadable: {e}"
    out = []
    names = list(ClientRecord.__dataclass_fields__)
    for i, raw in enumerate(lines):
        try:
            row = json.loads(raw)
            if type(row) is not dict or set(row) != set(names):
                return out, f"client.jsonl line {i}: fields differ from the record schema"
            out.append(ClientRecord(*ev.parse_fields(ClientRecord, [row[n] for n in names])))
        except ValueError as e:
            return out, f"client.jsonl line {i}: {e}"
    return out, None


def client_id_of(engine_id: str) -> str | None:
    """vLLM's chat endpoint names an engine request chatcmpl-<X-Request-Id>-<8 random characters>."""
    if not engine_id.startswith("chatcmpl-") or len(engine_id) < 19 or engine_id[-9] != "-":
        return None
    return engine_id[len("chatcmpl-"):-9]


def join(rep: Replay, clients: list[ClientRecord], item_sha: dict[int, str] | None) -> tuple[dict, list[str]]:
    problems = []
    by_client = {}
    for c in clients:
        if c.client_id in by_client:
            problems.append(f"client id {c.client_id} recorded twice")
        by_client[c.client_id] = c
    joined = {}
    for q in rep.requests.values():
        cid = client_id_of(q.engine_id)
        c = by_client.get(cid)
        if c is None:
            problems.append(f"engine request {q.engine_id} has no client record")
            continue
        if cid in joined.values():
            problems.append(f"client id {cid} joins two engine requests")
        joined[q.r] = cid
        if item_sha is not None and item_sha.get(c.item) != q.prompt_sha:
            problems.append(f"engine request {q.engine_id} prompt differs from corpus item {c.item}")
        if c.status == "ok" and c.response_id != f"chatcmpl-{cid}":
            problems.append(f"client {cid} response id {c.response_id} does not name its request")
        if c.status == "ok" and c.completion_tokens != q.num_output:
            problems.append(f"client {cid} usage {c.completion_tokens} tokens, engine delivered {q.num_output}")
    for cid, c in by_client.items():
        if cid not in joined.values() and c.status == "ok":
            problems.append(f"client {cid} completed with no engine request")
    return joined, problems


def pct(xs: list, q: float):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def median(xs: list):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def mean(xs: list):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def request_timing(c: ClientRecord) -> dict:
    ttft = None if c.t_first_ns is None else (c.t_first_ns - c.t_submit_ns) / 1e9
    tpot = None
    if c.t_first_ns is not None and c.t_last_ns is not None and (c.completion_tokens or 0) >= 2:
        tpot = (c.t_last_ns - c.t_first_ns) / 1e9 / c.completion_tokens
    return {"ttft_s": ttft, "tpot_s": tpot, "e2e_s": (c.t_end_ns - c.t_submit_ns) / 1e9}


def run_metrics(rep: Replay, clients: list[ClientRecord]) -> dict:
    submitted = len(clients)
    ok = [c for c in clients if c.status == "ok" and c.finish_reason in ("stop", "length")]
    failed = [c for c in clients if c.status in ("http_error", "error")]
    timing = [request_timing(c) for c in ok]
    eps = list(rep.episodes.values())
    resumed = [e for e in eps if e.recovery is not None]
    fractions = [e.fraction for e in resumed if e.fraction is not None]
    decs = rep.decisions
    preempted_requests = sum(1 for q in rep.requests.values() if q.preemptions)
    steps = rep.steps
    bins, _ = batch_distribution(steps)
    intervals = [ns for ns in _intervals_ns(steps) if ns is not None]
    span = None
    if clients:
        span = (max(c.t_end_ns for c in clients) - min(c.t_submit_ns for c in clients)) / 1e9
    out_tokens = sum(c.completion_tokens or 0 for c in ok)
    m = {
        "submitted": submitted, "completed": len(ok), "failed": len(failed),
        "unfinished": submitted - len(ok) - len(failed),
        "cap_hit_rate": (sum(c.finish_reason == "length" for c in ok) / len(ok)) if ok else None,
        "prompt_tokens_mean": mean([c.prompt_tokens for c in ok]),
        "output_tokens_mean": mean([c.completion_tokens for c in ok]),
        "output_tokens_median": median([c.completion_tokens for c in ok]),
        "preemptions": len(decs),
        "preemptions_per_100_submitted": 100 * len(decs) / submitted if submitted else None,
        "requests_preempted_frac": preempted_requests / submitted if submitted else None,
        "preemptions_per_preempted_request": len(decs) / preempted_requests if preempted_requests else None,
        "victim_least_tied_frac": (sum(any(c.chosen and c.least_tied for c in d.candidates) for d in decs)
                                   / len(decs)) if decs else None,
        "decisions_with_ties": sum(sum(c.least_tied for c in d.candidates) >= 2 for d in decs),
        "eligible_candidates_median": median([sum(c.eligible for c in d.candidates) for d in decs]),
        "stock_rule_violations": sum(not d.stock_rule_holds for d in decs),
        "episodes": len(eps), "episodes_resumed": len(resumed),
        "episodes_unresolved": sum(e.status == "unresolved" for e in eps),
        "recovery_fraction_median": median(fractions), "recovery_fraction_mean": mean(fractions),
        "recovery_fraction_min": min(fractions, default=None), "recovery_fraction_max": max(fractions, default=None),
        **{key: (sum(getattr(e.recovery, part) for e in resumed) if resumed else None) for key, part in (
            ("recovered_original_freed_tokens", "original_freed"), ("recovered_original_shared_tokens", "original_shared"),
            ("recovered_equivalent_tokens", "equivalent"), ("recovery_unknown_tokens", "unknown"),
            ("missing_tokens", "missing"))},
        "reprefill_tokens": sum(q.reprefill_tokens for q in rep.requests.values()),
        "discarded_tokens": sum(q.discarded_tokens for q in rep.requests.values()),
        "batch_size_time_weighted": time_weighted_batch(bins),
        "batch_size_decode_time_weighted": time_weighted_batch([b for b in bins if b.kind == "decode"]),
        "batch_interval_max_s": max(intervals) / 1e9 if intervals else None,
        "steps": len(steps),
        "steps_mixed": sum(s.kind == "mixed" for s in steps),
        "steps_prefill_only": sum(s.kind == "prefill" for s in steps),
        "steps_decode_only": sum(s.kind == "decode" for s in steps),
        "ttft_median_s": median([t["ttft_s"] for t in timing]), "ttft_p90_s": pct([t["ttft_s"] for t in timing], 0.9),
        "ttft_p99_s": pct([t["ttft_s"] for t in timing], 0.99),
        "tpot_median_s": median([t["tpot_s"] for t in timing]), "tpot_p90_s": pct([t["tpot_s"] for t in timing], 0.9),
        "e2e_median_s": median([t["e2e_s"] for t in timing]), "e2e_p99_s": pct([t["e2e_s"] for t in timing], 0.99),
        "throughput_interval_s": span,
        "output_tokens_per_s": out_tokens / span if span else None,
        "requests_per_s": len(ok) / span if span else None,
        "stalled_requests": sum(1 for c in clients if (c.max_gap_ns or 0) > STALL_NS),
        "alloc_failures": rep.alloc_failures, "hint_divergences": rep.hint_divergences,
        "works_unacknowledged": rep.works_unacknowledged,
        "snapshots_checked": rep.snapshots_checked, "digests_checked": rep.digests_checked,
    }
    return m


@dataclass
class RunHistory:
    attempt: str
    status: RunStatus
    reasons: list
    journal: ev.Journal
    replay: Replay | None
    clients: list
    joined: dict
    metrics: dict
    runner: dict
    distributions: Distributions | None = None


def load_run(directory: Path, item_sha: dict[int, str] | None = None) -> RunHistory:
    """Everything known about one attempt directory. Never raises on bad evidence; the status says
    complete only when the journal is sealed, replay and join pass, and every request finished."""
    reasons = []
    runner = {}
    if (directory / "attempt.json").exists():
        try:
            runner = json.loads((directory / "attempt.json").read_text())
        except (OSError, ValueError) as e:
            reasons.append(f"attempt.json unreadable: {e}")
        if type(runner) is not dict:
            reasons.append("attempt.json is not an object")
            runner = {}
    journal, rep = replay_attempt(directory)
    clients, client_problem = read_client(directory / "client.jsonl")
    if client_problem:
        reasons.append(client_problem)
    if isinstance(rep, Invalid):
        return RunHistory(directory.name, "invalid", [f"replay: {rep.reason} (seq {rep.seq})", *reasons], journal,
                          None, clients, {}, {}, runner)
    joined, join_problems = join(rep, clients, item_sha)
    reasons.extend(join_problems)
    metrics = run_metrics(rep, clients)
    status: RunStatus = "complete"
    if reasons:
        status = "invalid"
    else:
        if journal.status != "sealed":
            reasons.append(f"journal {journal.status}: {journal.reason}")
        if rep.end is not None and rep.end.unfinished:
            reasons.append(f"{rep.end.unfinished} engine requests unfinished at seal")
        if metrics["unfinished"] or metrics["failed"]:
            reasons.append(f"{metrics['unfinished']} unfinished and {metrics['failed']} failed client requests")
        if runner.get("outcome") not in (None, "complete", "validating"):
            reasons.append(f"runner outcome {runner.get('outcome')}: {runner.get('reason')}")
        expected = runner.get("requests")
        if expected is not None and metrics["submitted"] != expected:
            reasons.append(f"{metrics['submitted']} of {expected} requests submitted")
        if reasons:
            status = "incomplete"
    return RunHistory(directory.name, status, reasons, journal, rep, clients, joined, metrics, runner,
                      distributions(rep))
