# Search-G1 PPO safeguards

The user-approved three safeguards are integrated into the common8 actor loop.
This implementation does not change the learning rate, original masked-mean
loss denominator, GRPO advantages, reward, sampling temperature, reference KL,
batch schedule, or gradient clipping threshold.

* Dual-Clip: epsilon remains 0.2; c=3 is an explicit candidate. Positive and
  negative branches bound delta before exp; invalid masked positions are removed
  before subtraction. Effective invalid probabilities or advantages are rejected
  collectively before backward.
* Old-policy brake: four frozen trajectories per rank, selected across actor-token
  length strata, are checked before and after every mini. This is a fixed subset,
  not a complete-rollout check. Inputs/old probabilities/masks/advantages are hash
  bound and unchanged. k3 uses globally summed weighted numerator/denominator;
  negative-token quantiles are exact global order statistics. Negative-tail
  fractions use only negative-advantage mass/count. The weights preserve equal
  two-row micro-batch masked-mean semantics. The module's original train/eval mode
  is restored after monitoring.
* Finite arithmetic: invalid/over-bound KL samples reject before expm1 and report
  incomplete KL, never a clipped true KL. FP64 global gradient clipping preserves
  the original L2 formula and epsilon, measures the actual post-clip norm, and
  checks actual weights and Adam after every write. Durable write evidence is
  emitted immediately after Adam returns, before post-write finite checks.

All ranks bind the same full actor configuration before branching. A required
common8 run cannot disable safety. All four selected trajectories must have
positive effective length on every rank before any mini. A completed outer needs
20 actual writes and 40 ordered before/after monitor records per rank. A rejected
post-write mini retains its real write count and causes controlled failure; no
complete outer, fresh-rollout recovery or full-state rollback is claimed.

## Threshold admission

The recipe enables safety. Production admission requires
`BASE/logs/policy-safety-calibration.json`, bound to the actual run, source, model,
data and independently hashed initial probability/healthy short-segment proofs.
KL 0.01 and log bound 30 are explicit candidates. `negative_tail_limit` remains
unset until actual healthy calibration. CPU fixtures and synthetic native tests
are not eligible production proofs. A calibration producer and healthy Qwen
short-segment GPU/512-question EM/F1/action-rate acceptance remain required.

The original actor/reference/rollout probability gate is unchanged. As of native
diagnostic3251, all actual synchronized weights and prompt/position histories
match, actor/reference matches, but actor/rollout max error remains 1.550987,
above the original 0.05 gate. Fresh production updates remain zero. Safeguards
cannot substitute for this gate or repair already degraded weights.

## Verification

15 actual-target CPU/Gloo tests passed; scalar/hash evidence is recorded in
`reproduction/evidence/g1-common8-policy-safety-cpu.json`. Independent review
found no remaining static implementation blocker in its checked scope.
`test_policy_safety_native.py` is an isolated eight-GPU component fixture: actual
FSDP/Adam/actor loop with synthetic forward and explicitly replaced admission.
Native3252 completed successfully with all eight ranks: 20 normal writes and 40
checks per rank; rank7 invalid-old rejected before any write; post-write KL brake
retained exactly one write without a complete outer. Measured norms were about
2 before clipping and 0.999999523 after. Weights/Adam remained finite and old
identity unchanged. Scalar/hash proof is in
`reproduction/evidence/g1-common8-policy-safety-native.json`. It is not Qwen
training or healthy production calibration; its tiny threshold is a test trigger.

## Independently reconstructable global statistics

Later additive monitor fields preserve the decision and objective unchanged:
`weighted_k3_numerator`, `effective_weight_mass`,
`negative_effective_weight_mass`, and `tail_effective_weight_mass`. They are
already globally reduced. Identical records from eight ranks represent one
observation and must not be summed again. A finite numerator from an incomplete
KL estimate is only a diagnostic subtotal; it is not a valid full KL. Nonfinite
global totals produce null accumulators and collective rejection.

18 actual-target CPU/two-process Gloo tests passed in 10.626 seconds; hashes and
scope are in `g1-common8-policy-safety-cpu-accumulators.json`. The historical
native3252 proof remains bound to its frozen source; it does not prove these
later additive fields on eight GPUs. Actual healthy Qwen calibration and the
fixed512 EM/F1/effective-action assessment are still required.

## Pure calibration evidence verification

`safety_calibration_evidence.verify_segment` verifies complete eight-rank
20-write/40-check schedules, eachLR1e-6, frozen local identities, complete
globally reduced statistics, independent weighted-ratio arithmetic, and
unchanged-policy continuity between adjacent mini checks. Post-clip norms are
bounded by1 plus FP32 rounding epsilon; that tolerance is numerical, not a health
threshold. Real CPU monitor-to-worker-JSON-writer compatibility is tested.
The reader returns structural verification with `production_admission=False`.
It cannot authenticate physical execution, certify a healthy checkpoint,
choose thresholds or replace the remaining discovery/confirmation runner,
fixed512 EM/F1/actions and independent admission producer. Its tests are part
of the49-test target CPU proof; fabricated schedule fixtures are not real updates.
