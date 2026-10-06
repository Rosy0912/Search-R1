# Fresh200 controlled refusal state preservation

This is an evidence path for `tree_fresh200_five_leaf_v1`, not rollback,
checkpoint recovery, optimizer restart, probability acceptance, or training
completion. No loss, Tree leaf measure, minibatch denominator, or LR formula is
changed. Legacy profiles retain their scheduler/control behavior.

The actor resets an independent successful-return counter at update entry. It
increments immediately after `optimizer.step()` returns, before safety evidence
publication and rank audit. A failure later in publication, gradient audit,
post-write finite/KL checks, finish, or fresh scheduler accounting does not erase
the real returned writes. An optimizer exception before returning remains a
generic failure, whose internal partially applied state cannot be inferred from
this counter.

`PolicySafetyRefusal.matched_all_ranks` defaults to false. Only the already
synchronized runtime startup, published global brake, and fresh scheduler
refusals explicitly set it true. Only this typed true flag in the fresh profile
permits the worker snapshot collectives. Generic exceptions and manually raised
unmatched refusals propagate to the driver-only diagnostic path; they never
start a speculative worker collective after a rank failure.

Before the actor update, all ranks bind the driver context to the actual six
run/source environment keys, the fresh training profile, attempted/completed
outer, and a directory under this run. On a matched refusal each rank rechecks
the same context and attempts its local raw snapshot. Invalid/mismatched context
produces a common incomplete result without writing an untrusted path.

Snapshots contain actual parameter tensors, available private local shards,
gradients (including `None`), buffers, full local Adam state and parameter
groups/LR, scheduler state, Python/NumPy/Torch RNG (current CUDA device RNG for
NCCL workers), module training flags, and layout/offload metadata. Values are
copied before the worker's normal offload. Finite classification never prevents
preservation of nonfinite Adam, parameters, or gradients. All snapshots prohibit
automatic resume even when finite. No HF export, state restoration or executor
resume is attempted.

The driver supplies `tree_partial_context`:

```text
protocol: tree_partial_state_v1
identity: job_id, run_id, code_commit, config_sha256, data_identity,
          model_identity, training_protocol, attempted_outer, completed_outer
directory: <trainer.default_local_dir>/training_state/partial_outer_%06d
```

Each rank exclusively writes `worker-%02d.pt` and its SHA/size-bearing
`worker-%02d.json`. The receipt self SHA is carried by the return metadata, not
self-inserted in its JSON. Failed PT or receipt publication stays
`PARTIAL_WORKER_STATE_INCOMPLETE`; a PT already written before receipt IO failure
retains its actual path/hash in that incomplete result. Existing files are never
overwritten. The underlying same-world `atomic_save` helper syncs file bytes;
this implementation does not add a filesystem-crash or storage-latency claim.

Every rank gathers the direct receipts and returns identical
`meta_info['tree_partial_state']` with the original driver identity,
`worker_states`, actual scheduler advancement per rank, `outer_completed=false`
and `automatic_rollback=false`. Status is `PARTIAL_WORKER_STATE_SAVED` only if
all ranks saved; otherwise it is incomplete. Counts can differ after partial
progress and are recorded individually rather than averaged or zeroed. Worker
return occurs after the all-rank save attempt, allowing the driver's stop handler
to preserve sampler, controller, Tree RNG and consumed actor batch before
metric reduction or any completed-outer credit.

The fresh scheduler independently requires this actual counter and returned
metric to both equal 20 with zero skips on every rank. A scheduler/write-record
fault becomes a matched refusal. If the scheduler already advanced before the
fault, its actual new value is saved, while the driver still gives no completed
outer credit. First-outer LR zero can give real Adam calls with zero parameter
delta; the existing 57-outer schedule is unchanged.

Local verification uses the actual affected actor methods and worker method
extracted by AST, plus tiny real CPU Adam/LambdaLR and two-rank Gloo. Final nine
test methods passed, including six Gloo scenarios: update-return then publication
IO failure; post-write audit failure; zero-write pre-brake; one-rank snapshot
disk failure; nonfinite raw state; and wrong context. Per-rank returned envelopes
were byte-identical. Additional tests cover a nonzero-LR parameter change,
nonfinite Adam/gradient/local-shard/buffer preservation, unchanged RNG,
no-clobber, scheduler-advanced IO refusal, direct-counter/metric disagreement,
invalid preparation and exact fresh/legacy/unmatched worker control branches.

The first run's post-write audit fixture lacked `__file__` in its AST execution
namespace. That original failure log is retained. Supplying the original source
filename fixed the fixture; the affected scenario and final suite passed. This
was not a production-source failure or an FSDP result.

Real FSDP/NCCL/Ray, full model snapshots, shared-storage capacity/latency, Slurm
cleanup timing, production partial-refusal saves, corrected probability gates,
fresh calibration and 200-outers training remain unverified by these CPU tests.
The source must undergo the parent's independent review and a new clean freeze;
the old `1e06c6` source-bound dispatch helper cannot authorize this changed source.
