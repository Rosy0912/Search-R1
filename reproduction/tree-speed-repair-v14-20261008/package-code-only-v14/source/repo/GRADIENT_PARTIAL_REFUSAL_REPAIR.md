# F04: synchronized clipping refusal preserves actual partial worker state

The original second-mini nonfinite FSDP clipping path completed its bad-gradient
MAX reduction, then raised a generic `ValueError` before the actor's safety
checks. The worker only persists matched `PolicySafetyRefusal`. Both ranks could
therefore already have one successful Adam write, but return no worker snapshot.
The independent reviewer counterexample and our pre-fix regression preserve
this failure rather than treating the attempted outer as completed.

`clip_grad_norm64` now raises the `ValueError` subclass
`SynchronizedGradientNormRefusal` only from that completed bad-MAX numeric
branch. It marks `matched_all_ranks=True` only when the FSDP process group covers
the default worker world. The current actor worker constructs a one-dimensional
FSDP mesh spanning `dist.get_world_size()`; subgroup-size inequality remains
unmatched and cannot enter the worker's world-wide partial-state collectives.
There are no new collectives, changed reductions or clipping coefficients.

At both preclip and postclip measurement calls, an enabled actor safety runtime
converts only that exact exception type with the strict true marker into a
matched `PolicySafetyRefusal`, retaining the original exception as its cause.
The existing worker then persists raw actual state and returns the common
partial envelope. Generic `ValueError`, setup/dtype failures, failed collectives,
local non-FSDP nonfinite failures and unmatched subgroup refusals are unchanged.
With actor safety absent, the specialized refusal remains `ValueError`
compatible and does not authorize partial worker persistence.

CPU regression uses the actual clipping function and AST-executed actor
`_optimizer_step` and worker `update_actor`, real two-rank Gloo, tiny Adam,
synthetic FSDP object layout and a lazy-init bypass. On a rank-one NaN in the
second preclip or postclip, both ranks return the same saved partial envelope:
actual successful optimizer counter and Adam step are one, scheduler is zero,
changed parameters and nonfinite gradients are saved without rollback,
rank-one `state_finite=False`, and outer/resume/rollback credit remains false.
Only the fixture's temporary tiny CPU snapshots are loaded for byte/state
checks and removed. Additional local fixtures distinguish ordinary exceptions,
failed reduction and a synthetic subgroup-size marker. This is source/control
flow and CPU evidence, not real FSDP/NCCL/Ray/GPU acceptance or dispatch release.

The PPO loss, dual clip, KL checks, gradients, LR, scheduler, Tree sampling,
advantages/weights, configuration and successful-return counters are unchanged.
