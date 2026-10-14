# Native four-card formal audit failure 718650

The actual JXU allocation 718650 failed with exit 1:0 at 2026-10-14 22:20:47
Asia/Shanghai on ACD1-1. Its immutable source is
`5816a20f331e10aaff9bbb95c313611ead289fd7`, run ID
`c3856315d3f345d7920ebf5725a083b4`.

Five real actor updates and five question-head commits completed. Each rank
processed 640 trajectories with local mini32/micro4 and made 100 cumulative
Adam writes. HF435 and all four Adam/scheduler/RNG states, driver/candidate
stream, selector/head/replay state are saved at outer5. The head version is
`qprobe:5:5:21061c22ff64203f`. Formal outer6 was not started.

The failed reader hardcoded monitor weight mass16; the following formal reader
also hardcoded local micro2. Actual worker code divides each monitor micro's
token mask by its own mask sum and then sums over ranks. For 32 total monitor
trajectories and bound local micro4, the correct complete weight mass is 8.
The repair derives geometry from the bound run; it changes no PPO, reward,
normalization, guard threshold, original worker text or historical eight-card
calibration contract. Missing/bad native4 bindings and wrong mass still fail.

The source checkpoint was captured **before** dev512-after5. That evaluation
completed, but the subsequent acceptance failed before publication of any
completed-pair receipt or live post-evaluation worker/driver RNG boundary.
Do not fabricate a completed-pair status or claim bitwise restoration of that
lost boundary. CPU full SHA preservation is not actual CUDA/FSDP recovery.

`formal_audit_failure718650.py start --commit <published-sha>` makes one durable
claim and detaches a CPU-only preservation controller. Inspect an existing
claim/process/result after an unknown outcome; never replay it. It retains
large files remotely, makes independent small evidence copies and reports
actual full-read byte progress. No GPU job or training is submitted by it.

Recovery remains required: freshly verify the saved checkpoint and immutable
raw prefix, implement explicit native4 restoration of the real saved boundary,
verify actor/HF/Adam/scheduler/RNG/head/replay/selector/constructed-stream
readbacks, preserve the measured dev512 result and truthfully complete pair5.
Then continue outer6..200 with an authenticated source prefix and all five
checkpoint archives/final native four-rank restore/no201 acceptance. Do not
reset to official0 or regenerate the accepted frozen teacher5120 cache.
