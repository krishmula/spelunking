"""Manual review report for one run root: decision CSV, per-run HTML, configuration summary.

`render(root)` reads only the root's input files and writes `<root>/report/`. Output is byte for
byte deterministic: times are relative to each run's first journal event, iteration is sorted,
and floats use fixed formats.

annotations.tsv rows are `decision_key<TAB>request<TAB>why worth testing`, where decision_key is
`<attempt>:<decision>` (for example `r1.a1:0`) and request is the replay request number or the
client id. A first row starting with `decision_key` is a header and is skipped.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from e02 import history

ATTEMPT_FILES = ("attempt.json", "events.jsonl", "snapshots.jsonl", "client.jsonl")
RECOVERY_TOKEN_METRICS = ("recovered_original_freed_tokens", "recovered_original_shared_tokens",
                          "recovered_equivalent_tokens", "recovery_unknown_tokens", "missing_tokens")
QUEUE_HEAD = 16

COLUMNS = [
    "run_id", "attempt", "run_status", "target", "cap", "rep_seed", "decision", "group", "call", "t_rel_s",
    "requester", "needed", "free", "stock_rule_holds", "snapshot", "snapshot_link", "request", "client_id",
    "chosen", "eligible", "completed", "raw", "pending", "rank", "least_tied", "total_blocks", "shared_blocks",
    "reclaimable_blocks", "reserved_blocks", "shared_prefix_tokens", "sufficiency", "sufficiency_reason",
    "running_position", "first_admit_rel_s", "last_admit_rel_s", "preemptions_before", "reclaimed_blocks",
    "episode", "C", "H", "recovery_fraction", "original_freed", "original_shared", "equivalent", "unknown",
    "missing", "wait_s", "reprefill_tokens", "episode_status", "request_link",
]

LIMITS = (
    "Block generations and completed intervals are observer concepts. Fixtures on the real scheduler check "
    "them; snapshots do not. GPU re-prefill time is unavailable, so this report gives only completed "
    "re-prefill tokens. Alternative candidates are observations at a decision, not measured outcomes: "
    "choosing another victim would change every later allocation. Events within a run share one history, "
    "so many preemptions in one run are not independent samples."
)


@dataclass
class Attempt:
    name: str
    run_id: str
    k: int
    spec: dict
    meta: dict
    observed: bool
    hist: history.RunHistory | None = None
    snapshots: dict | None = None

    @property
    def status(self) -> str:
        if self.hist is None:
            return self.meta.get("outcome") or "unknown"
        if self.hist.status == "invalid":
            return "invalid"
        if self.meta.get("outcome") in ("failed", "invalid", "running", "validating"):
            return self.meta["outcome"]
        return self.hist.status

    @property
    def config(self) -> tuple:
        return (self.spec.get("target"), self.spec.get("cap"))

    @property
    def t0(self) -> int:
        events = self.hist.journal.events if self.hist else ()
        return events[0].t_ns if events else 0



def fmt(x, places: int = 6) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{places}f}"
    return str(x)


def rel(t: int | None, t0: int) -> str:
    return "" if t is None else f"{(t - t0) / 1e9:.6f}"


def fraction_text(ep: history.PreemptionEpisode) -> str:
    if ep.C == 0:
        return "undefined"
    if ep.recovery is None:
        return "unresolved"
    return f"{ep.recovery.H / ep.C:.4f}"


def esc(x) -> str:
    return html.escape(fmt(x) if not isinstance(x, str) else x)


def table(headers: list, rows: list, attrs: str = "") -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table{attrs}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


CSS = """
body{font-family:system-ui,sans-serif;margin:1.5em;color:#222}
table{border-collapse:collapse;margin:.5em 0 1.2em;font-size:13px}
th,td{border:1px solid #ccc;padding:2px 6px;text-align:right;vertical-align:top}
th{background:#f2f2f2;position:sticky;top:0}
td:first-child,th:first-child{text-align:left}
tr.chosen td{background:#ffe3c2;font-weight:600}
tr.tie td:first-child{border-left:4px solid #3a6fd8}
.status-complete{color:#176317}.status-incomplete,.status-running{color:#9a6300}
.status-invalid,.status-failed{color:#b00020}
.controls{margin:.5em 0}.controls label{margin-right:1em}
code{font-size:12px}
"""

JS = """
(function(){
const tb=document.querySelector('#decisions tbody');if(!tb)return;
const rows=Array.from(tb.rows);
const $=id=>document.getElementById(id);
function apply(){
 const d=$('f-decision').value,ch=$('f-chosen').checked,ti=$('f-ties').checked,su=$('f-suff').value;
 rows.forEach(r=>{r.hidden=!((d===''||r.dataset.decision===d)&&(!ch||r.dataset.chosen==='yes')
  &&(!ti||r.dataset.tied==='yes')&&(su===''||r.dataset.sufficiency===su));});
}
let state={key:'idx',dir:1};
function sortBy(key){
 const dir=(state.key===key)?-state.dir:(key==='idx'?1:-1);state={key:key,dir:dir};
 rows.sort((a,b)=>{
  if(key==='sufficiency'){
   const ua=a.dataset.sufficiency==='unknown',ub=b.dataset.sufficiency==='unknown';
   if(ua!==ub)return ua?1:-1;
  }
  const k=key==='sufficiency'?'suffrank':key;
  const va=Number(a.dataset[k]),vb=Number(b.dataset[k]);
  if(va!==vb)return dir*(va-vb);
  return Number(a.dataset.idx)-Number(b.dataset.idx);
 });
 rows.forEach(r=>tb.appendChild(r));
 $('sort-state').textContent=key==='idx'?'decision order':key+(dir<0?' descending':' ascending');
}
['f-decision','f-chosen','f-ties','f-suff'].forEach(id=>$(id).addEventListener('change',apply));
document.querySelectorAll('button[data-sort]').forEach(b=>b.addEventListener('click',()=>sortBy(b.dataset.sort)));
})();
"""


def page(title: str, body: str, script: bool = False) -> str:
    js = f"<script>{JS}</script>" if script else ""
    return (f"<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"><title>{esc(title)}</title>"
            f"<style>{CSS}</style></head><body>\n{body}\n{js}</body></html>\n")


def write_csv(path: Path, headers: list, rows: list) -> None:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(headers)
    w.writerows(rows)
    path.write_text(buf.getvalue())



class Inputs:
    """Every input file read, so the index footer can list its sha256."""

    def __init__(self, root: Path):
        self.root = root
        self.digests: dict[str, str] = {}

    def note(self, path: Path) -> None:
        if path.exists():
            self.digests[path.relative_to(self.root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()

    def text(self, path: Path) -> str:
        self.note(path)
        return path.read_text()


def load_attempts(root: Path, plan: dict, inputs: Inputs) -> list[Attempt]:
    specs = {s["run_id"]: s for s in plan.get("runs", [])}
    order = {s["run_id"]: i for i, s in enumerate(plan.get("runs", []))}
    item_sha = {int(k): v for k, v in plan["item_sha"].items()} if plan.get("item_sha") else None
    out = []
    attempts_dir = root / "attempts"
    dirs = sorted(p for p in attempts_dir.iterdir() if p.is_dir()) if attempts_dir.exists() else []
    for d in dirs:
        meta = json.loads(inputs.text(d / "attempt.json")) if (d / "attempt.json").exists() else {}
        base, _, k = d.name.rpartition(".a")
        run_id = meta.get("run_id") or base or d.name
        spec = specs.get(run_id, {})
        logging = meta.get("logging", spec.get("logging", True))
        a = Attempt(d.name, run_id, int(meta.get("attempt", k if k.isdigit() else 0)), spec, meta, bool(logging))
        if a.observed:
            for f in ATTEMPT_FILES[1:]:
                inputs.note(d / f)
            a.hist = history.load_run(d, item_sha)
            a.snapshots = a.hist.replay.snapshots if a.hist.replay is not None else {}
        out.append(a)
    by_name = {a.name: a for a in out}
    for a in out:
        if not a.spec and a.meta.get("replaces") in by_name:
            a.spec = by_name[a.meta["replaces"]].spec
    out.sort(key=lambda a: (order.get(a.run_id, len(order)), a.run_id, a.k, a.name))
    return out


def read_annotations(path: Path, inputs: Inputs) -> list[list[str]]:
    if not path.exists():
        return []
    rows = []
    for line in inputs.text(path).splitlines():
        if not line.strip():
            continue
        parts = (line.split("\t") + ["", "", ""])[:3]
        if parts[0] == "decision_key" and not rows:
            continue
        rows.append(parts)
    return rows



def client_of(a: Attempt, r: int) -> str:
    q = a.hist.replay.requests.get(r)
    return a.hist.joined.get(r) or (history.client_id_of(q.engine_id) if q else None) or ""


def decision_rows(a: Attempt) -> list[dict]:
    rep = a.hist.replay
    t0 = a.t0
    rows = []
    for dec in rep.decisions:
        ep = rep.episodes.get(dec.episode) if dec.episode else None
        for c in dec.candidates:
            row = {
                "run_id": a.run_id, "attempt": a.name, "run_status": a.status, "target": a.spec.get("target"),
                "cap": a.spec.get("cap"), "rep_seed": a.spec.get("rep_seed"), "decision": dec.d,
                "group": dec.group, "call": dec.call, "t_rel_s": rel(dec.t, t0), "requester": dec.requester,
                "needed": dec.needed, "free": dec.free, "stock_rule_holds": dec.stock_rule_holds,
                "snapshot": dec.snapshot, "snapshot_link": f"runs/{a.name}.html#snap-{dec.snapshot}",
                "request": c.r, "client_id": client_of(a, c.r), "chosen": c.chosen, "eligible": c.eligible,
                "completed": c.completed, "raw": c.raw, "pending": c.pending, "rank": c.rank,
                "least_tied": c.least_tied, "total_blocks": c.total_blocks, "shared_blocks": c.shared_blocks,
                "reclaimable_blocks": c.reclaimable_blocks, "reserved_blocks": c.reserved_blocks,
                "shared_prefix_tokens": c.shared_prefix_tokens, "sufficiency": c.sufficiency,
                "sufficiency_reason": c.sufficiency_reason, "running_position": c.running_position,
                "first_admit_rel_s": rel(c.first_admit_t, t0), "last_admit_rel_s": rel(c.last_admit_t, t0),
                "preemptions_before": c.preemptions_before,
                "request_link": f"runs/{a.name}.html#req-{c.r}",
            }
            if c.chosen:
                row["reclaimed_blocks"] = dec.reclaimed_blocks
                if ep is not None:
                    row.update(episode_fields(ep))
            rows.append({k: fmt(row.get(k)) for k in COLUMNS})
    return rows


def episode_fields(ep: history.PreemptionEpisode) -> dict:
    r = ep.recovery
    out = {"episode": ep.id, "C": ep.C, "recovery_fraction": fraction_text(ep),
           "reprefill_tokens": ep.reprefill_tokens, "episode_status": ep.status}
    if r is not None:
        out.update(H=r.H, original_freed=r.original_freed, original_shared=r.original_shared,
                   equivalent=r.equivalent, unknown=r.unknown, missing=r.missing,
                   wait_s=f"{(r.t_resume - ep.t) / 1e9:.6f}")
    return out


def run_page(a: Attempt, rows: list[dict]) -> str:
    h = a.hist
    parts = [f"<h1>Attempt {esc(a.name)}</h1>",
             f"<p>Run {esc(a.run_id)}, target {esc(a.spec.get('target'))}, cap {esc(a.spec.get('cap'))}, "
             f"rep seed {esc(a.spec.get('rep_seed'))}. Status: "
             f"<b class=\"status-{esc(a.status)}\">{esc(a.status)}</b>. "
             f"Runner outcome: {esc(a.meta.get('outcome'))}. Times are seconds from the run's first event.</p>"]
    reasons = list(h.reasons)
    if a.meta.get("reason"):
        reasons.append(f"runner: {a.meta['reason']}")
    if reasons:
        parts.append("<ul>" + "".join(f"<li>{esc(x)}</li>" for x in reasons) + "</ul>")
    if h.replay is None:
        parts.append("<p>Replay failed, so this attempt has no decisions or request histories.</p>")
        return page(a.name, "\n".join(parts))
    parts.append(decision_section(rows))
    parts.append(distribution_section(a))
    parts.append(snapshot_section(a))
    parts.append(request_section(a))
    return page(a.name, "\n".join(parts), script=True)


SHOWN = ["decision", "group", "t_rel_s", "request", "client_id", "chosen", "eligible", "completed", "raw",
         "pending", "rank", "least_tied", "total_blocks", "shared_blocks", "reclaimable_blocks",
         "shared_prefix_tokens", "sufficiency", "sufficiency_reason", "running_position", "preemptions_before",
         "reclaimed_blocks", "C", "H", "recovery_fraction", "episode_status"]


def decision_section(rows: list[dict]) -> str:
    decisions = sorted({int(r["decision"]) for r in rows})
    options = "".join(f"<option value=\"{d}\">{d}</option>" for d in decisions)
    controls = (
        "<div class=\"controls\">"
        f"<label>Decision <select id=\"f-decision\"><option value=\"\">all</option>{options}</select></label>"
        "<label><input type=\"checkbox\" id=\"f-chosen\"> chosen victim only</label>"
        "<label><input type=\"checkbox\" id=\"f-ties\"> least-computed ties only</label>"
        "<label>Sufficiency <select id=\"f-suff\"><option value=\"\">all</option><option>yes</option>"
        "<option>no</option><option>unknown</option></select></label></div>"
        "<div class=\"controls\">Sort: <button data-sort=\"idx\">decision order</button> "
        "<button data-sort=\"completed\">completed</button> <button data-sort=\"reclaimable\">reclaimable</button> "
        "<button data-sort=\"prefix\">shared prefix</button> <button data-sort=\"sufficiency\">sufficiency</button> "
        "<span id=\"sort-state\">decision order</span>. Unknown sufficiency always sorts last.</div>")
    body = []
    for i, r in enumerate(rows):
        cls = " ".join(x for x in ("chosen" if r["chosen"] == "yes" else "",
                                   "tie" if r["least_tied"] == "yes" else "") if x)
        mark = ", ".join(x for x in ("victim" if r["chosen"] == "yes" else "",
                                     "least tie" if r["least_tied"] == "yes" else "") if x)
        cells = [f"<a href=\"#snap-{r['snapshot']}\">snap {r['snapshot']}</a>",
                 f"<a href=\"#req-{r['request']}\">req {r['request']}</a>", esc(mark)]
        cells += [esc(r[k]) for k in SHOWN]
        data = (f" data-idx=\"{i}\" data-decision=\"{r['decision']}\" data-chosen=\"{r['chosen']}\""
                f" data-tied=\"{r['least_tied']}\" data-sufficiency=\"{r['sufficiency']}\""
                f" data-completed=\"{r['completed']}\" data-reclaimable=\"{r['reclaimable_blocks']}\""
                f" data-prefix=\"{r['shared_prefix_tokens']}\" data-suffrank=\"{1 if r['sufficiency'] == 'yes' else 0}\"")
        body.append(f"<tr class=\"{cls}\"{data}>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    head = "".join(f"<th>{esc(h)}</th>" for h in ["snapshot", "request", "mark", *SHOWN])
    return (f"<h2>Decisions</h2><p>{len(decisions)} decisions, {len(rows)} candidate rows. The chosen victim "
            "is shaded; every least-computed tie has a blue left edge.</p>" + controls +
            f"<table id=\"decisions\"><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>")


def snapshot_section(a: Attempt) -> str:
    rep = a.hist.replay
    used = sorted({d.snapshot for d in rep.decisions})
    out = ["<h2>Decision snapshots</h2>"]
    for i in used:
        entry = a.snapshots.get(i)
        if entry is None:
            out.append(f"<details id=\"snap-{i}\"><summary>Snapshot {i}: missing</summary></details>")
            continue
        rec = entry[1]
        refs = {row[0]: row[1] for row in rec["blocks"]}
        decs = [d.d for d in rep.decisions if d.snapshot == i]
        head = rec["queue"][:QUEUE_HEAD]
        rows = []
        for row in rec["requests"]:
            r, status, tbl, computed, in_flight, stale, num_tokens, num_output, pre, can_free = row
            owned = " ".join(f"{b}({refs.get(b, 0)})" for b in tbl)
            rows.append([f"<a href=\"#req-{r}\">{r}</a>", esc(client_of(a, r)), esc(status), esc(computed),
                         esc(in_flight), esc(stale), esc(num_tokens), esc(num_output), esc(pre), esc(bool(can_free)),
                         f"<code>{esc(owned)}</code>"])
        blocks = [[esc(b), esc(ref), esc(nk)] for b, ref, nk, _ in rec["blocks"]]
        out.append(
            f"<details id=\"snap-{i}\"><summary>Snapshot {i} (decision {', '.join(map(str, decs))}, "
            f"call {rec['call']}, victim {esc(rec['victim'])}, free {rec['free']})</summary>"
            f"<p>Free blocks: {rec['free']}. Free queue head: <code>{esc(' '.join(map(str, head)))}"
            f"{' ...' if len(rec['queue']) > QUEUE_HEAD else ''}</code>. Running order: "
            f"<code>{esc(' '.join(map(str, rec['running'])))}</code>. Waiting: "
            f"<code>{esc(' '.join(map(str, rec['waiting'])))}</code>.</p>"
            + table(["request", "client", "status", "computed", "in flight", "stale", "tokens", "outputs",
                     "preemptions", "can release", "owned blocks (ref)"], rows)
            + "<p>Referenced or cached blocks:</p>" + table(["block", "ref", "hash keys"], blocks)
            + "</details>")
    return "\n".join(out)


def request_section(a: Attempt) -> str:
    rep, t0 = a.hist.replay, a.t0
    clients = {c.client_id: c for c in a.hist.clients}
    out = ["<h2>Requests</h2>"]
    for r in sorted(rep.requests):
        q = rep.requests[r]
        cid = client_of(a, r)
        c = clients.get(cid)
        resumes = [rep.episodes[e].recovery.t_resume for e in q.episodes if rep.episodes[e].recovery]
        lines = [f"Engine id <code>{esc(q.engine_id)}</code>, client {esc(cid or 'none')}.",
                 f"Prompt {q.prompt_len} tokens, max {q.max_tokens}. Arrival {rel(q.t_arrival, t0)} s.",
                 f"First admission {rel(q.first_admit_t, t0) or 'never'} s; resumptions "
                 f"{', '.join(rel(t, t0) for t in resumes) or 'none'}.",
                 f"Final status {esc(q.status)}, finished at {rel(q.finish_t, t0) or 'not finished'} s, "
                 f"{q.num_output} output tokens delivered, {q.reprefill_tokens} re-prefill tokens, "
                 f"{q.discarded_tokens} discarded tokens."]
        if c is not None:
            t = history.request_timing(c)
            lines.append(f"Client: status {esc(c.status)}, finish {esc(c.finish_reason or 'none')}, "
                         f"TTFT {fmt(t['ttft_s']) or 'undefined'} s, TPOT {fmt(t['tpot_s']) or 'undefined'} s, "
                         f"{fmt(c.completion_tokens) or 'unknown'} output tokens.")
        eps = []
        for e in q.episodes:
            ep = rep.episodes[e]
            rc = ep.recovery
            eps.append([esc(ep.id), f"<a href=\"#snap-{ep.snapshot}\">{ep.decision}</a>", rel(ep.t, t0),
                        esc(ep.C), esc(ep.raw), esc(ep.pending_tokens), esc(ep.reclaimed_blocks),
                        esc(rc.H if rc else ""), esc(rc.original_freed if rc else ""),
                        esc(rc.original_shared if rc else ""), esc(rc.equivalent if rc else ""),
                        esc(rc.unknown if rc else ""), esc(rc.missing if rc else ""), esc(fraction_text(ep)),
                        f"{(rc.t_resume - ep.t) / 1e9:.6f}" if rc else "", esc(len(ep.lookups)),
                        esc(rc.intervening_alloc_blocks if rc else ""),
                        f"{(ep.regained_t - ep.t) / 1e9:.6f}" if ep.regained_t is not None else "",
                        esc(ep.reprefill_tokens), esc(ep.status)])
        ep_table = table(["episode", "decision", "t", "C", "raw", "pending", "reclaimed", "H", "orig freed",
                          "orig shared", "equivalent", "unknown", "missing", "H/C", "wait s", "lookups",
                          "intervening alloc blocks", "time to regain C s", "re-prefill", "status"], eps) \
            if eps else "<p>Never preempted.</p>"
        out.append(f"<section id=\"req-{r}\"><h3>Request {r}</h3><p>{' '.join(lines)}</p>{ep_table}</section>")
    return "\n".join(out)



DISTRIBUTION_COLUMNS = ["attempt", "run_id", "target", "cap", "distribution", "key", "value"]
QUANTILES = (("min", 0.0), ("p25", 0.25), ("median", 0.5), ("p75", 0.75), ("max", 1.0))


def value_lists(d: history.Distributions) -> dict[str, tuple]:
    """Each run's per-event distributions, by name. Events in one run share one history."""
    out = {f"candidate_{k}": v for k, v in sorted(d.candidate.items())}
    out["recovery_fraction"] = d.recovery_fraction
    out["episode_reprefill_tokens"] = d.episode_reprefill_tokens
    return out


def distribution_rows(a: Attempt) -> list[list]:
    d = a.hist.distributions
    if d is None:
        return []
    head = [a.name, a.run_id, fmt(a.spec.get("target")), fmt(a.spec.get("cap"))]
    rows = []
    for b in d.batch:
        rows.append([*head, "batch_seconds", f"{b.kind}:{b.batch}", fmt(b.seconds)])
        rows.append([*head, "batch_steps", f"{b.kind}:{b.batch}", fmt(b.steps)])
    for name, values in value_lists(d).items():
        rows.extend([*head, name, str(i), fmt(v)] for i, v in enumerate(values))
    return rows


def quantile_cells(values) -> list[str]:
    return [esc(len(values))] + [esc(num(history.pct(list(values), q))) for _, q in QUANTILES]


def distribution_section(a: Attempt) -> str:
    d = a.hist.distributions
    if d is None:
        return "<h2>Distributions</h2><p>Replay failed, so this attempt has no distributions.</p>"
    total = sum(b.seconds for b in d.batch)
    batch = [[esc(b.kind), esc(b.batch), esc(b.steps), esc(f"{b.seconds:.6f}"),
              esc(f"{b.seconds / total:.4f}" if total else "undefined")] for b in d.batch]
    spread = [[esc(name), *quantile_cells(values)] for name, values in value_lists(d).items()]
    return ("<h2>Distributions</h2>"
            f"<p>Batch-size time is a {esc(d.batch_basis)}. Final step interval: {esc(d.batch_final_interval)}.</p>"
            + table(["forward pass kind", "batch size", "steps", "seconds", "share of time"], batch)
            + "<p>Per-event distributions within this run. Its events share one history, so they are not "
              "independent samples.</p>"
            + table(["distribution", "n", *(k for k, _ in QUANTILES)], spread))


def config_distributions(runs: list) -> str:
    """One row per complete run and distribution; runs are never pooled into one sample."""
    rows = []
    for a in runs:
        d = a.hist.distributions
        if d is None:
            continue
        for name, values in value_lists(d).items():
            rows.append([esc(a.name), esc(name), *quantile_cells(values)])
    if not rows:
        return "<p>No per-run distributions.</p>"
    return ("<p>Per-run distributions. Each row summarizes one run's events; compare runs, not pooled events.</p>"
            + table(["attempt", "distribution", "n", *(k for k, _ in QUANTILES)], rows))


def stats(values: list) -> dict:
    xs = [v for v in values if v is not None]
    return {
        "n": len(xs),
        "mean": statistics.fmean(xs) if xs else None,
        "sd": statistics.stdev(xs) if len(xs) >= 2 else None,
        "median": statistics.median(xs) if xs else None,
        "min": min(xs) if xs else None,
        "max": max(xs) if xs else None,
    }


def num(x) -> str:
    if x is None:
        return "undefined"
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def is_tail(metric: str) -> bool:
    return "_p9" in metric


def metric_value(h: history.RunHistory, metric: str):
    if metric in RECOVERY_TOKEN_METRICS and not h.metrics.get("episodes_resumed"):
        return None
    return h.metrics.get(metric)


def config_blocks(configs: list, members: dict) -> tuple[list[str], list[list]]:
    html_parts, summary = [], []
    for cfg in configs:
        runs = members.get(cfg, [])
        target, cap = cfg
        html_parts.append(f"<h3 id=\"cfg-{target}-{cap}\">Target {esc(target)}, cap {esc(cap)}</h3>")
        html_parts.append(f"<p>{len(runs)} complete attempts enter these aggregates: "
                          f"{esc(', '.join(a.name for a in runs) or 'none')}.</p>")
        metrics = []
        for a in runs:
            for m in a.hist.metrics:
                if m not in metrics:
                    metrics.append(m)
        rows = []
        for m in metrics:
            values = [metric_value(a.hist, m) for a in runs]
            s = stats(values)
            agg = ["not aggregated"] * 3 if is_tail(m) else [num(s["mean"]), num(s["sd"]), num(s["median"])]
            rng = f"{num(s['min'])} to {num(s['max'])}" if s["n"] else "undefined"
            rows.append([esc(m), *(esc(num(v)) for v in values), *(esc(x) for x in agg), esc(rng), esc(s["n"])])
            summary.append([target, cap, m, ";".join(num(v) for v in values), *agg, num(s["min"]), num(s["max"]),
                            s["n"]])
        html_parts.append(table(["metric", *(a.name for a in runs), "mean", "sample SD", "median", "range", "n"],
                                rows))
        pooled = {"ttft_s": [], "tpot_s": []}
        for a in runs:
            for c in a.hist.clients:
                if c.status == "ok" and c.finish_reason in ("stop", "length"):
                    t = history.request_timing(c)
                    for k in pooled:
                        if t[k] is not None:
                            pooled[k].append(t[k])
        prow = []
        for k, xs in pooled.items():
            s = stats(xs)
            p90, p99 = history.pct(xs, 0.9), history.pct(xs, 0.99)
            prow.append([esc(k), esc(len(xs)), esc(num(s["median"])), esc(num(p90)), esc(num(p99)),
                         esc(num(s["max"]))])
            summary.append([target, cap, f"pooled_{k}", "", num(s["mean"]), num(s["sd"]), num(s["median"]),
                            num(s["min"]), num(s["max"]), s["n"]])
        html_parts.append("<p>Pooled request-level distribution across the complete attempts above. Each run's "
                          "own tails are in the metric table; they are not averaged into a configuration "
                          "tail.</p>")
        html_parts.append(table(["measure", "requests", "median", "p90", "p99", "max"], prow))
        html_parts.append(config_distributions(runs))
    return html_parts, summary


def status_table(attempts: list, aggregated: set) -> str:
    replaced_by = {}
    for a in attempts:
        if a.meta.get("replaces"):
            replaced_by[a.meta["replaces"]] = a.name
    rows = []
    for a in attempts:
        if not a.observed:
            continue
        reasons = list(a.hist.reasons)
        if a.meta.get("reason"):
            reasons.append(f"runner: {a.meta['reason']}")
        rows.append([f"<a href=\"runs/{esc(a.name)}.html\">{esc(a.name)}</a>", esc(a.run_id),
                     esc(a.spec.get("target")), esc(a.spec.get("cap")), esc(a.spec.get("rep_seed")),
                     esc(a.spec.get("round")), esc(a.meta.get("outcome")),
                     f"<span class=\"status-{esc(a.status)}\">{esc(a.status)}</span>",
                     esc(a.name in aggregated), esc(a.meta.get("replaces")), esc(a.meta.get("replacement_reason")),
                     esc(replaced_by.get(a.name)), esc("; ".join(reasons))])
    return table(["attempt", "run", "target", "cap", "rep seed", "round", "runner outcome", "run status",
                  "in aggregates", "replaces", "replacement reason", "replaced by", "reasons"], rows)


def unobserved_table(attempts: list) -> str:
    rows = [[esc(a.name), esc(a.run_id), esc(a.spec.get("target")), esc(a.spec.get("cap")),
             esc(a.spec.get("rep_seed")), esc(a.meta.get("outcome")), esc(a.meta.get("reason")),
             esc(a.meta.get("requests"))] for a in attempts if not a.observed]
    if not rows:
        return "<p>No unobserved attempts.</p>"
    return table(["attempt", "run", "target", "cap", "rep seed", "runner outcome", "reason", "requests"], rows)


def examples_section(annotations: list, rows_by_attempt: dict) -> str:
    if not annotations:
        return "<p>No annotations.tsv in this root.</p>"
    index = {}
    for name, rows in rows_by_attempt.items():
        for r in rows:
            index[(f"{name}:{r['decision']}", r["request"])] = r
            index[(f"{name}:{r['decision']}", r["client_id"])] = r
    out = []
    for key, request, why in annotations:
        r = index.get((key, request))
        if r is None:
            out.append([esc(key), esc(request), esc(why), "no matching decision row"] + [""] * 7)
            continue
        out.append([esc(key), f"<a href=\"{esc(r['request_link'])}\">{esc(request)}</a>", esc(why),
                    f"<a href=\"{esc(r['snapshot_link'])}\">snap {esc(r['snapshot'])}</a>",
                    *(esc(r[k]) for k in ("chosen", "least_tied", "completed", "reclaimable_blocks",
                                          "shared_prefix_tokens", "sufficiency", "recovery_fraction"))])
    return table(["decision", "request", "why worth testing", "snapshot", "chosen", "least tie", "completed",
                  "reclaimable", "shared prefix", "sufficiency", "recovery fraction"], out) \
        + "<p>The full decision table stays in decisions.csv and on each run page.</p>"


def render(root: Path) -> Path:
    root = Path(root)
    inputs = Inputs(root)
    plan_path = root / "manifest.json" if (root / "manifest.json").exists() else root / "plan.json"
    plan = json.loads(inputs.text(plan_path))
    attempts = load_attempts(root, plan, inputs)
    annotations = read_annotations(root / "annotations.tsv", inputs)

    out = root / "report"
    (out / "runs").mkdir(parents=True, exist_ok=True)

    replaced = {a.meta["replaces"] for a in attempts if a.meta.get("replaces")}
    rows_by_attempt: dict[str, list[dict]] = {}
    all_rows = []
    for a in attempts:
        if not a.observed:
            continue
        rows = decision_rows(a) if a.hist.replay is not None else []
        rows_by_attempt[a.name] = rows
        all_rows.extend(rows)
        (out / "runs" / f"{a.name}.html").write_text(run_page(a, rows))
    write_csv(out / "decisions.csv", COLUMNS, [[r[k] for k in COLUMNS] for r in all_rows])
    write_csv(out / "distributions.csv", DISTRIBUTION_COLUMNS,
              [row for a in attempts if a.observed and a.hist.replay is not None for row in distribution_rows(a)])

    configs = sorted({(s.get("target"), s.get("cap")) for s in plan.get("runs", [])} |
                     {a.config for a in attempts if a.observed},
                     key=lambda c: (c[0] is None, c[0] or 0, c[1] is None, c[1] or 0))
    members: dict[tuple, list[Attempt]] = {}
    for a in attempts:
        if a.observed and a.status == "complete" and a.name not in replaced:
            members.setdefault(a.config, []).append(a)
    aggregated = {a.name for runs in members.values() for a in runs}
    cfg_html, summary = config_blocks(configs, members)
    write_csv(out / "summary.csv", ["target", "cap", "metric", "values", "mean", "sd", "median", "min", "max", "n"],
              summary)

    counts: dict[str, int] = {}
    for a in attempts:
        key = a.status if a.observed else "unobserved"
        counts[key] = counts.get(key, 0) + 1
    footer = table(["input file", "sha256"], [[esc(p), f"<code>{d}</code>"] for p, d in sorted(inputs.digests.items())])
    body = "\n".join([
        f"<h1>E02 report: {esc(plan.get('root_id'))}</h1>",
        f"<p>Kind {esc(plan.get('kind'))}, corpus digest <code>{esc(plan.get('corpus_digest'))}</code>. "
        f"Attempts by status: {esc(', '.join(f'{k} {v}' for k, v in sorted(counts.items())) or 'none')}. "
        f"{len(all_rows)} candidate rows in <a href=\"decisions.csv\">decisions.csv</a>; configuration "
        "aggregates in <a href=\"summary.csv\">summary.csv</a>; per-run distributions in "
        "<a href=\"distributions.csv\">distributions.csv</a>.</p>",
        "<h2>Run status</h2><p>Every observed attempt is listed. Only complete attempts that no other attempt "
        "replaces enter the aggregates.</p>", status_table(attempts, aggregated),
        "<h2>Unobserved attempts</h2><p>These cost-control runs had logging off. They have no event history "
        "and never enter the aggregates.</p>", unobserved_table(attempts),
        "<h2>Configurations</h2><p>Each run value comes from one complete attempt. Sample SD needs two runs. "
        "Recovery is undefined, not zero, when no episode resumed. Percentile tails are shown per run and not "
        "averaged.</p>", *cfg_html,
        "<h2>Selected examples</h2>", examples_section(annotations, rows_by_attempt),
        f"<h2>Limits</h2><p>{esc(LIMITS)}</p>",
        "<footer><h2>Inputs</h2>", footer, "</footer>",
    ])
    (out / "index.html").write_text(page(f"E02 report {plan.get('root_id')}", body))
    return out
