The failure occurs before actor optimization: the immutable ablation config
places training state in `base/checkpoints`, while the original worker validator
requires that state beneath `SEARCH_R1_BASELINE_OUTPUT`'s parent (`base/output`).

Use `bootstrap.install_into` as the ablation150 import callback for
`search_r1.baselines.partial_worker_state`, in the entry and every Ray worker.
The callback retains the native validator and replaces its one root expression
with a frozen receipt/config/run-bound exact checkpoint namespace. Native source
files remain unchanged. An inactive ablation delegates to the original validator.

Also use `bootstrap.install_driver_into` for `search_r1.baselines.tree_partial_state`.
For the explicit zero-write replay of attempted outer1/completed0/attempt1, it
chooses `base/checkpoints/replay-<current-output-attempt>/training_state/<name>`.
The saved batch therefore keeps its real consumed cursor and attempt identity,
and a second failure cannot overwrite the original evidence. The worker accepts
only that exact receipt-bound replay path or the normal exact checkpoint path.

This does not change the existing synchronized numerical guard, optimizer,
scheduler, PPO objective, thresholds, sampling, budget, or evaluation schedule.
Programming, provenance, IO, and state contamination faults still stop. Evidence
from a failed attempt is immutable; a recovery must use a fresh attempt identity.

The subsequent native-only rolling snapshots, sparse HF/native keypoints, and
post-evaluation snapshots use `trainer.default_local_dir` without the incompatible
audit-root check. Native resume binds every member to this group's checkpoint
manifest and still requires all eight states. This namespace repair alone does
not make the incomplete zero-write partial manifest resumable.
