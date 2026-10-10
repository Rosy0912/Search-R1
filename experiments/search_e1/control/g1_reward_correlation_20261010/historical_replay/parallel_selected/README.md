# Parallel selected historical TRAIN reward reconstruction

Latest human authorization permits sharing the original Search-G1 training and
auxiliary GPUs0–3/5–6. Devices4/7 stay excluded. Evaluate only1,20,40,60,80,100,120,140.
Step1 is retained from the accepted old diagnostic, not recomputed. Root handles
stopping only its exact bound dense diagnostic; the old source and partial/accepted
results are preserved. This new source does not stop or modify any original job.

Six independent workers process whole original2560-trajectory iterations:
GPU0:[60],1:[80],2:[100],3:[120],5:[20,40],6:[140]. Each worker has an explicit
ordered unique `selected_steps` subset, separate output/session and per-GPU lock.
No batch is split between GPUs or merged with another iteration. Scientific
micro4/HF-SDPA/BF16/4096/left padding/last-hidden protocols remain unchanged.
Only used historical encoder versions are full SHA-streamed and loaded; unused
versions are neither loaded nor claimed fresh by that worker. Every value is
`R_reconstructed`; original online per-row R was not saved.

The old step1 summary lacks attempt_id. Its accepted2560 rows are streamed to
verify run/step/unique native attempt/probe/permutation IDs and fresh SHA, then
joined to the committed original ledger. The old summary is never modified.
Each new per-step row and summary schema stays compatible with the existing
independent statistics reader. A worker's terminal summary covers only its assigned
points, plus a reference to retained1. Global completion/Pearson uses an explicit
file map and counts1 once: eight points /20,480 individual C/R pairs. Do not add
worker summaries or describe this selected scope as dense TRAIN1–150.

Original ownership is strict per candidate:0–3 permit only bound `trainer`
ledger processes;5 only bound `retriever`;6 bound `retriever`/`probe-server`.
Every PID/starttick/SID/UID/normalized command SHA must match the actual original
run ledger. The exact new worker may additionally occupy its configured GPU.
Unknown processes HOLD this own diagnostic; it never sends a signal. Separate
PID/SID and per-GPU locks prevent joining original process ledgers or duplicate
allocation. Source/config/run/UUID identities are freshly checked before Torch.

Training sharing uses initial actual free≥32768MiB, own cap12032MiB and running
actual free≥6144MiB; auxiliary workers use cap16384MiB/running free≥8192MiB.
A bound original sampled-used peak proof checks the nominal training budget:
peak + own cap + floor≤physical total. This is sampled historical used memory,
not an instantaneous maximum or historical direct-free measurement. Current
actual direct free is sampled every2s. A no-op model pre-forward guard checks
observer failure before each original micro4 and aborts only the own diagnostic;
it changes no forward input, output, kwargs, sampling or model RNG. A cap/floor
HOLD is preserved and remaining points can be reassigned under separate authorization
and configuration. No original trainer/probe/retriever is stopped or changed.

All weights, traces, features, per-row pairs, cache/tmp/memory journals and outputs
remain on the original Linux node under the new remote cohort directory. Mac
retains code/configuration/docs and compact summaries only. CPU preparation and
fixtures start no GPU. After actual launch verify two complete selected iterations,
exact native FP32 reward mean within2e−6, printed min/max/EM consistency, and real
ownership/memory health. Aggregate agreement does not prove historical rowwise
bitwise identity. Finish only after all seven new points plus retained1 are
accepted and the independent exact20,480-row statistics audit passes.
