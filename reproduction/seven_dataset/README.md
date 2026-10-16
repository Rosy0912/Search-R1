# Frozen milestone seven-dataset evaluation

This entry evaluates the authorized original and recursive ablation milestones
on the fixed 51,713-question split. It never calls `fit`, restores Adam, or
modifies a training process. The original-200/H100 legacy final launcher is not
used. Training-only ablation and schedule fields are first validated against
the archived ablation contract and then projected to the common original
configuration. All models use the native one-trajectory greedy search evaluator,
with unchanged search/response limits, retrieval, token and answer semantics.

`plan` only writes a model-specific plan. `entry --validate-only` checks actual
HF hashes, archived milestone metadata, configuration and data hashes and all
51,713 unique ordered question identities without inference. Old native payloads
may have been pruned; their metadata is not represented as restorable state.

The submitter must verify the published source snapshot, exact existing scheduler
job and model-identity submission intent before acquiring an independent eight-GPU
allocation. `launch` acquires idle GPU leases. A matching unique submission intent
is required before `entry` can run. Existing training and retrieval allocations
must remain untouched. A failed or uncertain submission must be inspected, never
blindly repeated. Original-200 native completed evaluations must be reused.

The evaluator saves each question's identity, prediction, ground truth, EM and
semantic response spans, and fsyncs every completed batch. Acceptance requires
all seven source counts, all question IDs in their frozen order, independently
recomputed EM, unchanged model hashes and eight distinct frozen worker GPU UUIDs.
The result records each dataset's correct/count/EM, equal-dataset and all-question
EM, prediction hashes and raw emitted tokens. Scheduler GPU cost is reconciled
after the job exits; the running summary only states an inference-time lower bound.

Validation: thirteen contract tests, including exact PT/Slurm allocation binding,
plus native-environment full input validation against retained depth-2 and
original round-50 exports. GPU execution remains subject to
the first actual evaluator job's runtime acceptance checks.

Checkpoint exports may serialize tokenizer files differently. Inference uses a
separate hardlinked model view: trained checkpoint weights and the exact tokenizer
files from the archived training model path, checked against the training raw-token
SHA contract. Exported checkpoint bytes remain intact. A known FAILED allocation
that produced no predictions can be replaced only under its own recovery namespace,
with exact failed-allocation and submission bindings; the prior output is retained.
