# Pure OPSD E1 trainer/state gate implementation

2026-10-04. Local candidate only, in `work/opsd-e1-300`; no commit, deployment, GPU job or receipt was created by this implementation task. File SHA256 inventory: `trainer-gate-files.json`. CPU raw output: `trainer-gate-cpu.txt` (39 passed, 6.36s; includes 20 checkpoint-gate cases and legacy trainer regressions).

## Behavior

Enabled only by `baselines.opsd_e1.enabled=true`. Disabled legacy trainer counter/save/validation behavior remains in its existing branch. E1 requires 300 actual outer updates, 57 absolute outer warmup, pure OPSD/no reference KL, unchanged token-mean distillation, CUDA FP32 eager teacher, user-context hints and no component clipping. This implementation does not modify actor/distillation math.

The training loader uses the shared `StatefulSampler`, zero loader workers and fixed 169028 questions. Each successful outer commits 512 original questions after the eight-rank update audit. Every rank must report 320 trajectories, 20 actual optimizer calls and zero skipped minis, finite loss/gradient and valid masks, teacher frozen/CPU-idle storage/FP32/eager, and native default allocator. An empty global mini still follows the existing skip behavior; its skipped count rejects the outer rather than silently adding an Adam clock tick.

Initial and every-50 validation require 512 evaluated questions; stage64 is separately evaluated at step5 and each50 without affecting the objective. HF plus full actor/driver state saves occur at 5 and every25 through300. Final51713 runs only after actual300 and logs against updated step300. There is no sentinel301 update.

At step5 the workers perform their SHARDED_STATE_DICT full-state same-object disk roundtrip RPC. Driver sampler, RNG and algorithm state also actually reload and compare. The manifest requires all eight native rank files/reports, eight distinct worker PIDs, native DTensor entries, scheduler epoch5 and optimizer state, HF index/all shards/tokenizer/config, exact launch data/model/tool/environment/import/source records and resolved config. The gate pauses inside the fifth loop body, before the sixth `next()`; driver, sampler iterator and worker objects remain alive.

Only an external RSA-SHA256 review signature can release the pause. Training has no signing implementation or private key. The receipt binds the fresh nonce, job, PID, source, config, horizon300/warmup57 and every checkpoint byte. It also binds independent review/resources/retrieval/HF reload/rank audit records, and a byte-exact prefix snapshot of the real train log containing precisely updates1..5 and all eight-rank audits. Appended Ray output is allowed; changing the captured prefix, a stale receipt, wrong identity, symlink or changed checkpoint rejects release.

The import gate verifies the actual seven core module paths, source hashes and live compiled code structure plus source filename/line. Real seven-module CPU imports passed; a replaced live function with unchanged source bytes is rejected. Raw `marshal` byte equality was deliberately avoided because equivalent code can serialize with different reference/intern layouts.

## Explicit recovery boundary

`CheckpointGate.restore_driver(state_record, authorization_record)` is an explicit, separately signed driver recovery primitive. Its exact authorization schema binds checkpoint SHA/digest, original PID/job, destination PID/UID/job, source/config and step. It restores sampler cursor/permutation, CPU Python/NumPy/Torch RNG, algorithm state, actual outer count, next global step and epoch and verifies actual state afterward. The normal first-five flow never calls it, and a same-PID step-five release is rejected as a restart authorization.

This function alone does **not** verify worker recovery, resume the distributed training loop automatically, or approve a new Slurm job. It returns `worker_recovery_verified=false`. Full end-to-end new-process recovery still requires external independently signed destination/rank bindings, the worker recovery RPCs, import/data/model/config gates, and actual native eight-rank state/RNG/sixth-update proof. The shared full_state/worker implementation has separate CPU recovery tests; those do not constitute GPU recovery acceptance.

## Meaningful CPU evidence and limits

Tests exercise actual RSA signatures, wrong signed identities, modified checkpoint/log prefix/symlink rejection, waiting before sixth fetch, actual driver RNG/sampler disk roundtrip, authorized recovery and wrong destination/receipt-type rejection, fixed20/nonempty audit, an extracted actual trainer tail through300, native report completeness with fixture-only ranks, duplicate/nonfinite JSON/path escape, live code substitution and actual seven-module import closure. The native rank/HF fixture test intentionally uses mock files; it is not HF or FSDP GPU acceptance.

Pending before any real release: clean pinned commit/full runtime binding, real official data/model assets, actual eight-rank first5 with20 updates each, native DTensor full-state roundtrip, independent fresh HF reload, real costs/local retrieval and a human-authorized external independent receipt. No receipt has been fabricated and no GPU result is claimed here.

## Authorized follow-up delta: full imported source proof (candidate)

Root authorized a minimal integration delta after the original four-file freeze. The driver now reads the externally SHA-pinned closure and requires its `modules['reproduction.baselines.e1_import_bootstrap']` record to name the exact owned-repository checker source. It hashes one byte buffer and executes exactly that buffer, avoiding stale pyc imports, then invokes the full already-loaded module/function checker. The seven critical-object checks remain an additional independent check.

`init_workers` propagates `{checker_path,checker_sha256,closure_path,closure_sha256}` into `actor.e1_import_checker` before resource-pool creation. The worker-state author has implemented matching rank-local capture checks. CPU targeted regression at this intermediate delta: 41 passed/6.46s (`trainer-gate-delta-cpu.txt`), including changed-source/symlink rejection and executing the actual init_workers prefix to prove pins arrive before pool creation.

Intermediate full-import integration is **not yet accepted**: the profile checker was being hardened for unknown project code filenames/properties and its first real collect comparison rejected an equivalent code object (`verify_rsa_signature`); the author was notified of likely noncanonical frozenset constant representation. This failure is preserved in `driver-full-import-integration-cpu.txt`. Final checker bytes, real full-closure collection, driver/rank integration and a new SHA freeze are still required. The previous seven-module result is not retroactively described as this stronger proof.

### Delta integration result and new four-file freeze

After the profile author normalized frozenset constants and tightened project alias/property handling, actual CPU collection and the fresh-byte driver verifier passed on67 modules, followed by the independent seven-critical-object check. `driver-full-import-integration-cpu.txt` records that result; the first failure remains separately in `driver-full-import-integration-first-failed.txt`.

The worker author's `verify_rank_import_checker` was also invoked in this actual CPU process against the same four propagated pins. It verified67 modules and1069 live functions and logged actual PID6390/UID501 (`rank-full-import-integration-cpu.txt`). Rank0 was only the CPU interface argument; this is not a real GPU rank or Slurm job proof. The source inventory is now deterministic complete repository Python bytes for identity, while process-dependent already-loaded inventories are separately checked rather than compared as restored state. The previously observed cross-process dynamic-inventory failure is not erased.

Combined driver/gate, bootstrap, full-state and three legacy regression test files: **97 passed in16.34s**, stdout `trainer-gate-integration-cpu.txt`. New four-file freeze: `trainer-gate-delta-files.json` (gate `e1ba71ec34b3889dcd6964f3c07d0bd266f5847f3039226191f8b87a38a910cf`; trainer `32cbbae68f7b76adb94c34a307ae95e892d4443b781df9bbf940ef2841c213a8`). These tests are local candidate integration; no receipt/release, training, remote deployment, native FSDP or eight-rank GPU recovery was run.
