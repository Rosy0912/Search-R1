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

The original actor/reference/rollout probability gate is unchanged. Actual native
3266 passed on all eight ranks and the original64 questions: actor/reference
max0, actor/rollout max0.00055694580078125, below the original0.05. Scoped FP32
actor/reference/FSDP/rollout, actual internal activation/logits and vLLM KV dtype
were observed; no global monkeypatch or relaxed tolerance was used. This is
initial single-action probability acceptance, not agent EM, coexistence, a
2560-trajectory peak or healthy training. Fresh production updates remain zero.
Earlier3251 probability failures remain immutable evidence.

## Independent calibration phases and production binding

Discovery and confirmation are distinct fresh official-model children of the
still-unstarted production parent. Each preserves the original301 horizon and
performs exactly one complete512-question/G5 outer: actual20 Adam writes and40
fixed-monitor checks per rank, a complete finite canonical checkpoint1, and
paired evaluation of the original ordered512 validation rows before/after.
They retain duplicate canonical question occurrences and original GRPO groups.
No final51713 evaluation or terminal probe refit is dispatched for these children.

Discovery observes the negative tail without treating an invented value as a
production threshold. KL0.01, log bound30 and c3 remain explicit candidates;
finite rejection remains enforced. Confirmation requires a threshold choice and
health rule frozen from the actual discovery evidence before it starts. Both
children must use the same actual initial heads, encoder and reward mathematics;
authenticated adoption preserves raw probe bytes and labels and publishes a new
target-run version. It copies no accepted marker. Actual server unload/reload,
HTTP, probability and coexistence admissions must still be performed locally.

The bounded runtime produces measurements with production_admission=False.
The independent schema2 production producer re-reads actual source/data/model
identity, ordered monitor logs, actual optimizer-state summaries and process
birth receipts, complete checkpoint bytes/delta, paired512 metrics and owned
cleanup. It interprets only the explicit frozen health rule. The production
artifact retains the parent identity and separately identifies the two actual
child runs/jobs; their20 writes are never relabeled as parent writes. Production
actor workers verify the artifact read-only with a matched collective decision.
Missing or damaged evidence, legacy schema or missing source descriptors reject
before production writes. CPU fixtures do not qualify as healthy admission.

Validation records strict-answer EM and normalized alias F1. `valid_action_rate`
counts recognized actions, including a recognized final search that the driver
does not execute; executed logical search calls and question search rate are
recorded separately. Neither metric is silently renamed as effective search.
The added F1/action statistics do not change the training reward.

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
choose thresholds or replace actual discovery/confirmation GPU execution,
fixed512 EM/F1/actions and independent admission verification. Its tests are part
of the49-test target CPU proof; fabricated schedule fixtures are not real updates.
