# Fix h3run: OOM (ran out of memory)

## Issue
Compute job `h3run` was OOM-killed — it exceeded its 48.0 GB memory cap (systemd MemoryMax), so every ticket waiting on its results is blocked. This is fixable: either cut the peak memory (e.g. micro-batch the log-probs/backward so the full [rows, T, vocab] tensor is never held) or raise the job line's `(mem:)` cap if the peak is genuinely needed.

## Evidence
Exit: `oom-kill`, code TERM; memory cap 48.0 GB.
Failing command: `bash scripts/gpu_job.sh`
Last console output (the UI does not show this):
  froze memory -> runs/20261013-035328-B-local-suite-s1-h3base/memory_frozen.json
[20261013-035328-B-local-suite-s1-h3base] TEST in-distribution on 5: ['enc-hex', 'cls-caesar', 'xor-single', 'for-grep', 'rev-logic']
  indist enc-hex      solved=True turns=3
  indist cls-caesar   solved=False turns=15
  indist xor-single   solved=False turns=15
  indist for-grep     solved=False turns=15
  indist rev-logic    solved=True turns=4
[20261013-035328-B-local-suite-s1-h3base] TEST transfer on 5: ['enc-base32', 'cls-atbash', 'xor-repeat', 'for-split', 'rev-arith']
  transfer enc-base32   solved=True turns=4
  transfer cls-atbash   solved=True turns=3
  transfer xor-repeat   solved=False turns=15
  transfer for-split    solved=True turns=3
  transfer rev-arith    solved=False turns=15
[20261013-035328-B-local-suite-s1-h3base] DONE -> runs/20261013-035328-B-local-suite-s1-h3base
Full log: /home/kevin/Documents/coders/loop/logs/jobs/mem-vs-ft-ctf-h3run.log

## Done when
Peak RSS measured and recorded, and the run completes (or `--steps 2` finishes) within the job's `(mem:)` cap. If you raise the cap instead of cutting memory, say why and record the peak.

## Stop when
After two attempts with no new evidence, record the blocker and stop — the signature then escalates to a human
card. Reproduce cheaply first (tiny config / small input), not by re-running the whole job.

## Plan
1. Reproduce the failure with the smallest/fastest invocation.
2. Fix the cause (trim memory / fix the command); add a fast check.
3. Make sure the fix reaches the job's branch so its retry picks it up.
