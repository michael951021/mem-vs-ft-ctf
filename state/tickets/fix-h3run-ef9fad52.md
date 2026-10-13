# Fix h3run: non-zero exit (exit-code)

## Issue
Compute job `h3run` exited non-zero (`exit-code`, exit 1) with no Python traceback — a failing command, not a crash in the training code. Blocks its dependents.

## Evidence
Exit: `exit-code`, code 1; memory cap 64.0 GB; peak 0.1 GB (systemd MemoryPeak, host-side).
Failing command: `bash scripts/gpu_job.sh`
Last console output (the UI does not show this):
  indist xor-single   solved=False turns=15
  indist for-grep     solved=False turns=15
  indist rev-logic    solved=True turns=4
[20261013-131825-B-local-suite-s1-h3base] TEST transfer on 5: ['enc-base32', 'cls-atbash', 'xor-repeat', 'for-split', 'rev-arith']
  transfer enc-base32   solved=True turns=4
  transfer cls-atbash   solved=True turns=3
  transfer xor-repeat   solved=False turns=15
  transfer for-split    solved=True turns=3
  transfer rev-arith    solved=False turns=15
[20261013-131825-B-local-suite-s1-h3base] DONE -> runs/20261013-131825-B-local-suite-s1-h3base
Auto-merging results/redaction_audit.json
CONFLICT (content): Merge conflict in results/redaction_audit.json
Automatic merge failed; fix conflicts and then commit the result.
error: job worktree /home/kevin/Documents/coders/loop/work/mem-vs-ft-ctf/job-h3run cannot take master (merge conflict): resolve it there
Full log: /home/kevin/Documents/coders/loop/logs/jobs/mem-vs-ft-ctf-h3run.log

## Done when
The command exits 0 on the real inputs, with a fast check that covers the fixed path.

## Stop when
After two attempts with no new evidence, record the blocker and stop — the signature then escalates to a human
card. Reproduce cheaply first (tiny config / small input), not by re-running the whole job.

## Plan
1. Reproduce the failure with the smallest/fastest invocation.
2. Fix the cause (trim memory / fix the command); add a fast check.
3. Make sure the fix reaches the job's branch so its retry picks it up.
