# NOTES — current state

## Task: fe0dada2d73a (per-cell pilot isolation + checkpoint) — done
scripts/cell_runner.py (cell-peak = child ru_maxrss) + run_pilot.sh
checkpoint store .tools/pilot_ckpt/$TAG ($key.done marker; CKPT_DIR=""
disables; MOCK=1 offline). Logs .agent-evidence/pilot-ckpt-{prefix,final,
run1..3}.
- Open resolved: canonical h3run set ran under isolation; est_gpu_hours
  placeholder replaced by measured 0.4375 (train_meta.json).

## Task: fix-h3run-ef9fad52 (h3run non-zero exit) — done
gpu_job.sh: _on_exit captures rc before killing HB; tree_rss_kb heartbeat.
Logs gpu-job-exit-{prefix,final} + -regressions. h3run then completed
(A–D 20261013-1755..1810; E/F 20261014-2305).

## Task: h3report (write up H3 in REPORT.md) — done
- Added §5.4 H3 between the auto block and §5.3: six arms on matched 7B
  base, 140 attempts, success/H1/H1b/H2/efficiency tables + caveats;
  stale H3 sentences fixed in §5.1/§5.3/§7/§9.2. Every number traces to
  summary_h3.json, train_meta.json, manifest.json, run config.yaml.
- Test scripts/tests/test_h3_report.py: Part A = raw attempts recomputed
  vs summary; Part B = verbatim rows + numeric-token sweep. Logs
  h3report-acceptance-{prefix (expected FAIL: no §5.4), final1, final PASS}.
- Findings: E's SFT corpus = 64 records from 1 source challenge
  (enc-base64); decontam dropped 8 rev-logic trajectories (eval leak);
  the 2026-10-13 build swept ~8 superseded runs -> corpus mixes code
  versions. E measured 0.4375 GPU-hours; the 1146.88 est is an uncalibrated
  placeholder, disclosed, not used.
- render_report --inject: auto-block run index now lists h3 runs (generator
  globs runs/); 27B numbers unchanged.

## Conventions
- Loop jobs run "no agent" via lib/jobs.py; shared binds: .tools, .venv-gpu,
  finetune/adapters, finetune/data, ~/.cache/huggingface (53GB). Committed 27B
  *-pilot artifacts never touched. Job log tee'd to .tools/h3run.joblog.
- Tests: plain python modules, `python3 -m <pkg>.tests.<name>` (no pytest).
- Ollama in agent sandbox: no model served; use $OLLAMA_BASE_URL or
  $OLLAMA_HOST. Direct curl to 127.0.0.1:11434 is hook-blocked.
