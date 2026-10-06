# Fresh200 opportunities, actual writes, and continuation

This implementation preserves the original fresh200 authorization timestamp
1791275767, the human token-KL update 1791282908, and the later human counting
update 1791283983 as separate proof/config fields. Token-KL is 0.2; the negative
joint-action tail remains a separate guard. GiGPO objective, groups, masks,
weights, gamma, omega, rollout geometry and learning-rate formula are unchanged.

Each newly fetched 512-question attempt has 20 optimizer opportunities. An
attempt counts as a complete outer only when all 20 opportunities were processed,
at least one actual Adam call succeeded, and remaining opportunities were not
abandoned. Only this condition advances the scheduler once. The target is 200
such completed outers; 20 Adam calls and 160 backward microbatches per outer are
the no-skip reference, never a validator of actual writes.

| Example | Processed | Actual Adam writes | Counted complete | Scheduler advance |
| --- | ---: | ---: | --- | ---: |
| One numerical opportunity skipped, others succeed | 20 | 19 | yes | 1 |
| All opportunities safely skipped | 20 | 0 | no | 0 |
| KL/tail abandons remaining rollout after a written step | 7 | 3 | no | 0 |

Only a known typed refusal at its original matched all-rank boundary enters
continuation. A pre-write refusal clears pending gradients and skips that
opportunity. A post-write refusal retains the actual written model/Adam. KL/tail
abandons remaining rollout reuse and fetches a fresh batch with newly computed
old log probabilities and GiGPO groups. Unknown forward, optimizer, FSDP, Ray,
collective or probe failures propagate. Written nonfinite model/Adam fails closed.
No rollback is performed. No rejected gradient is resumed.

`begin_attempt(trainer, batch)` binds actual fetched sampler epoch/cursor/order
SHA and a unique monotonically increasing attempt ID. `consume_attempt(trainer,
actor_output)` requires all eight real returned non-tensor rows under
`fresh200_attempt_receipts`; DataProto metadata alone retains only rank zero.
Every row records processed opportunities, successful writes, skipped
opportunities, actual successful backward microbatches, countable/abandoned
decision, actual cumulative Adam total and actual scheduler position. Rank/PID
coverage and common counters must agree. `record_worker_attempt(worker, receipt)`
is the actual update_actor scheduler/worker-ledger completion seam.

The driver ledger uses schema `FRESH200_OPPORTUNITIES_AND_ACTUAL_WRITES_V1` and
stores completed outers, attempts, skipped attempts, writes within completed
outers, retained writes in uncounted attempts, total writes, successful backward
microbatches, consumed questions, full attempt history, current attempt, and
actual exhausted-epoch tail records. Consumed questions are 512 times attempts;
sampler cursor is the actual current epoch cursor. Exhausted drop-last tails are
recorded separately. A new epoch uses the sampler's actual next permutation.

All partial states and rollout traces use unique attempt and opportunity/refusal
identities. Preclear snapshots retain actual model/Adam/RNG and pending gradients.
After all ranks verify gradients cleared and unchanged finite model/Adam, the
postclear state binds that immutable tensor payload by SHA and stores actual
postclear RNG/runtime metadata. `materialize_continuation_state(path, sha256)`
hashes and parses each same regular no-follow file descriptor, validates stable
file identity, reconstructs the evidence state and removes pending gradients.
These states are not complete checkpoints and cannot be automatically resumed.
The same live workers continue at the next opportunity or fresh batch.

For a roughly 3B-parameter FP32 actor, a preclear all-rank snapshot can contain
about 24 GB of model plus duplicated local shards, 24 GB of initialized Adam
moments, and up to 12 GB of pending gradients: roughly 60 GB per refusal before
serialization overhead. Twenty refusals can therefore approach 1.2 TB, plus
monitor/driver evidence. Postclear immutable references avoid a second full
tensor set. Further deduplication is deferred. Any write failure is a matched
fail-closed error; disk exhaustion cannot become a safe-skip loop.

Complete checkpoints at completed outers 5/50/100/150/200 save actual worker and
driver ledgers with model/Adam/scheduler/RNG/data/GiGPO state. Reload validates
actual Adam totals from the ledger, not 20 times completed outers, and the real
epoch/cursor/permutation, not 512 times completed outers. The first-five same-PID
acceptance gate runs before fetching completed outer six. HF semantic loader
helpers remain unchanged. CPU draft fixtures and GPU execution are separate;
these implementation changes and their new fixtures have not been executed.
