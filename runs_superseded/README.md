Superseded H3 (matched 7B base) run directories, moved out of `runs/` on 2026-10-14 so that
`scripts.analyze --glob '*-h3*'` aggregates exactly one run per (arm, seed).

h3run was retried ~8 times on 2026-10-12/13 (OOM kills, a numpy/pyarrow ABI break, then the
per-arm process isolation fix). Each try re-ran arms A-D from scratch and left its run dirs in
`runs/`, so the first summary_h3.json pooled 2-8 copies per cell (n=85 for A/B vs 55 for C/D) from
different code versions. The last try (20261013-1755..1810) is the only complete A-D x {s0,s1} set
on the final code, and is the canonical one kept in `runs/`. The empty E/F dirs here are from the
failed Ollama import (relative FROM path parsed as a registry host); h3serve re-runs E/F.
