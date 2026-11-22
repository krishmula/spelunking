# Pass 2: claudeAgent claude-opus-5-5@medium (condensed; full reply in the T3 delegated task r05-pass2-opus)

Tools: WebSearch, WebFetch, Bash (gh), Read. No files written. No pass-1 kill reversed.

## C5': NARROW to C5'' (gap confidence medium at most)
- Shortest rejection: Spills Secrets already extracts datastore contents and names user-derived datastores
  [L184 §5]; `cache_salt` is the fix [L188] [L195]. Sticks for the defence; only partly for the measurement, since
  no one tests an online co-tenant-filled cache.
- Combination (survives, sharper): pass 1 was wrong that prefix-cache attacks leak only the attacker's input. vLLM's
  prefix cache stores generated-token blocks (`kv_cache_manager.py` `cache_blocks`), but only reachable via the exact
  prefix from position 0 (chained hashes, salted first block). The suffix tree matches any context of up to 24 tokens
  (`max_tree_depth=24`) anywhere in any cached output. Position-free matching is the new constraint.
- Simplicity (Open): baselines `max_cached_requests=0`, tenant salt, and `--stream-interval` > 1 (Spills Secrets:
  20-step aggregation still leaves 9-52% accuracy).
- Scale (Open): default `stream_interval=1`, but `RequestOutputCollector` merges deltas when the API server is slow
  (`output_processor.py:51-73`). Snowflake: "production-ready at all concurrency levels" [L127]; RFC #51788 GPU
  index acceptance 1.53 to 2.33 from c1 to c256, CPU tree flat [L238].
- Oracle (Open; pass 1 wrong): no minimum count, `match_len` starts at 1, so a unique attacker context drafts a
  single-occurrence victim continuation at probability 1.0. Real limits: greedy target agreement; the guess cannot
  be literal in the attacker prompt (local tree wins ties); common contexts split probability; each probe inserts
  itself (false positives on repeats). Bound uncomputed.
- Relevance (Narrow): prefix-cache salt bypass CVE-2026-105752 rated Low (CVSS 3.1) sets the severity ceiling.
- Scoop: low. No suffix-cache privacy issue in vLLM, Arctic, SGLang, llama.cpp, TRT-LLM.
- C5'' question: (1) what fraction of a co-tenant's output spans can an attacker confirm, bounded by target
  agreement, local masking, count-ratio gating, self-insertion, frontend merging; (2) which isolation scope (key,
  org, salt, stream_interval k) removes it and at what acceptance cost.
- Kill condition: salting keeps more than 95% of acceptance and extraction matches the static-datastore case.
- New sources: CVE-2026-105752; Early Bird 2409.20002v5; RFC 9842; VUSec Dedup Est Machina Returns; llama.cpp
  #27852; vLLM code `output_processor.py:51-73`, `scheduler.py:214`, `kv_cache_manager.py`.
- Compute: Air offline screen about 1-2 h (Qwen3-4B 4-bit greedy, Arctic CPU cache replay); live under 10 Colab
  GPU-hours (2-3x uncertain).

## C2': KILL (simplicity, combination, relevance)
- Closed-form crossover; re-prefill on D reintroduces the interference PD removes. vLLM push connector blog
  (2026-10-14) shows transfer cost falling fast. No vLLM or SGLang issue asks for separate-drafter PD.

## C9': KILL as standalone
- SGLang reworking the EMA controller: #37274 (merged, title only), #28045 cost-aware, #31716 context-aware. Residue:
  adversarial acceptance collapse [L187] [L251] dragging the batch average for co-tenants; could ride on C5''.
