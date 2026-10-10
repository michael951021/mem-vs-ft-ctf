# Staging the GPU-heavy jobs on the-box agent hub

The study's heavy arms (the matched-base **A–D re-run on a 7–14B model** and the
**QLoRA arms E/F**) are GPU-bound. They are staged as JSON specs in
`jobqueue/jobs/` and run by `jobqueue/runner.py` on the **host** when GPU headroom
exists. This file explains how that maps onto the-box hub
(`http://the-box:8765/`) and why it is a host runner, not a sandbox loop.

## Why host-run, not a pool loop
The hub's `pool` (`loop/lib/pool.py`) dispatches **sandboxed Claude-on-Ollama**
coding loops: each worker sees only its project dir at `/work`, with
`--unshare-all` and inference over a bound `ollama.sock`. Real GPU **training**
(torch/bitsandbytes on `/dev/nvidia*`) is not exposed in that sandbox — which is
why the sibling `loopgym` project runs its GPU training **host-only**
(`scripts/gpu_run.sh … --dry-run` in the sandbox just prints the plan). QLoRA
here follows the same convention: the sandbox can dry-run it; the real run is on
the host.

## Run the queue (host, when a GPU is free)
```bash
cd ~/Documents/coders/mem-vs-ft-ctf
.venv/bin/python -m jobqueue.runner plan     # topological order + ETAs
.venv/bin/python -m jobqueue.runner list     # readiness incl. live GPU free-MB
.venv/bin/python -m jobqueue.runner run      # runs ready jobs; GPU-gated per spec
```
Each job declares `needs_gpu_free_mb`; the runner refuses to start it until
`nvidia-smi` reports that much free on some card, so nothing fights the live
Ollama server. Freeing the GPU typically means pausing Ollama during the QLoRA
window (the agent is denied from doing this itself — see `DECISIONS.md` D8 — so
it is an operator step).

## Optional: let the hub trigger it in rotation
To have the hub start the queue when a slot frees, register a **real** loop
project (the pool skips symlinks and names containing `smoke`):
```bash
cd ~/Documents/coders/loop
./agent new mem-vs-ft-ctf-eval
# put a single milestone in projects/mem-vs-ft-ctf-eval/TODO.md:
#   - [ ] Run the staged GPU queue: from the repo, `.venv/bin/python -m jobqueue.runner run`
#         (the harness must be reachable from the worker — bind it in .agent-shared or
#          clone it into the project; inference-only arms can use the ollama socket).
```
Because the worker's `/work` is isolated, the harness must be made reachable
(bind-mount via `.agent-shared`, or clone the repo into the project). For the
QLoRA step specifically, prefer the host runner above — the pool sandbox has no
GPU device.

## Job graph (see `jobqueue/jobs/*.json`)
```
pull-base-7b → pilot-7b-ABCD → build-sft → qlora-E → import-E-ollama → eval-E ┐
                                                                     └→ eval-F ┴→ analyze-all
```
`eval-F` needs `--load-memory` pointed at the frozen ACE playbook produced by the
7B arm-D pilot (`runs/<7B-D-s0>/memory_frozen.json`); fill that path in
`jobqueue/jobs/70_eval_F.json` after `pilot-7b-ABCD` completes.
