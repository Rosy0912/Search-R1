# Independent aligned EM200 deployment review

Date:2026-10-09. Reviewer: `/root/alignment_actor_review`.

**Result: PASS_SOURCE_DEPLOYMENT_REVIEW_NOT_TRAINING_RELEASE. No outstanding must-fix in the current reviewed deployment scope.** Training source remains fixed at `080ebbc46b89d61567cd33e1a5e2f1b40afd8b84`; this review did not change it. The earlier installer findings below were corrected by the owner and rechecked against the final byte identities listed below.

This approves the reviewed source-only installer, bound tool installation/transfer and CPU-observation entry within the task's authorized scope. It is not evidence that remote installation or target CPU tests ran, and grants no Slurm submission, GPU training or first-five release approval. No remote operation was performed by this reviewer.

## Resolved must-fix findings

1. **Child symlink escape.** Earlier `prepare` accepted `incoming`/`logs` directory links and followed existing artifact links. Current `safe_dir` requires real canonical owned directories. ROOT and ancestor canonicality are checked before creation, and marker, upload parts, assembled parts, bundle and completed-record links reject. The added assembled-part check also prevents bypassing the upload guard by calling `setup` directly.
2. **Incomplete completed-record replay.** Earlier same-hash replay checked only source HEAD/clean status and returned a prior success record despite missing or changed environment/model topology. Current replay validates task/root/scope, complete owned directory topology, deliberate dependency/model links, all three interpreter links, and copied pyvenv/asset JSON hashes before returning the matching source/bundle record. Missing or changed installation rejects without repair/overwrite.
3. **Tool-install completion admission.** The added installer now requires a regular non-linked completed record with exact task/root/source/scope, verified own topology and exact080 clean HEAD before writing bound tools. It rejects absolute/traversing/deeper-than-three-part paths, linked destination/parent directories and changed existing bytes; unchanged same-SHA files are reusable.
4. **Unknown transfer outcomes.** The client now rejects any existing action request/response/transport artifact before writing, creates requests exclusively and stops on a failed transport. An unknown previous result cannot silently resend or overwrite its original request.
5. **CPU target identity.** The new CPU entry now requires Linux `h100-5`, queries the actual new-prefix Python subprocess and requires its prefix to equal NEWROOT/env and its host/platform to be h100-5/linux. The observation records actual Python identity and host; it does not infer target identity from a path or caller assertion.

## Local counterexample validation

Executed the current installer functions only against temporary local fixture directories with mocked Linux host/mount/free-space and mocked Git queries. No external command was permitted in the harness. The official-model path was mapped to a temporary fixture. **25 original installer checks passed** (the final installer retains those tested functions byte-for-byte; reconstruction SHA is recorded):

- Unchanged same-hash completed read-back and repeated same-hash part upload succeed.
- Different-hash upload, unknown unmarked ROOT, ROOT/ancestor symlinks, incoming/logs links, linked marker/bundle/record, linked upload/assembled parts and missing parts reject.
- Partial setup with an existing checkout and no completed record rejects before clone/overwrite.
- Missing environment, external checkout link, changed dependency/model/interpreter link, changed copied pyvenv/asset JSON, and wrong completed-record task/root/scope reject.
- An external sentinel remained unchanged in every case.

The added tool installer passed **16** local path/completion/source-state counterexamples, including same-hash reuse, changed contents/hash, absolute/traversing paths, destination/parent links, missing/linked/empty/wrong-identity completion records and changed/dirty Git state. These tool checks mock the already-tested host/topology checks. The actual client `call` function passed **7** checks with mocked transport: fresh action, five existing evidence types, and failure followed by attempted resend; original evidence remained intact. Total: **48 local checks**.

Detailed case results and their exact reviewed source hashes are in `DEPLOYMENT-REVIEW-TESTS.json`. These fixtures verify guards and replay logic, not actual Git clone/fetch, NFS durability, live dependency contents, official model bytes or GPU readiness.

## Installer behavior and limits

- NEWROOT is exactly `/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009`. Admission checks h100-5/Linux, expected NFS4 source and shared free-space threshold. Live mount/quota observations were not repeated by this review.
- Source assembly checks supplied part/whole-bundle bytes; clone uses `--no-hardlinks --no-checkout` from old60, fetches the incremental bundle and checks out exact080 clean. Old source HEAD is checked before/after. No command modifies the old checkout or global environment.
- Own lexical environment and mutable output directories live under NEWROOT. Dependencies and the official-model mirror are referenced through prescribed links; the installer copies no old run/checkpoint, Adam, scheduler, RNG or acceptance receipt. It performs no model load, GPU allocation, Slurm submit or training.
- Completed replay verifies filesystem topology and copied configuration hashes. Official model/dependency contents still require actual runtime/preflight validation; the installer does not claim those checks. Marker/completion writes lack explicit fsync publication. An interrupted/unreadable completion remains fail-closed and requires inspection rather than automatic overwrite.

## Additional tool scope

`run-target-cpu.py` checks exact080 clean source, Linux/h100-5 and actual own lexical Python prefix. Its23 test suites run with empty CUDA visibility, offline model settings, one OMP thread and bytecode writes disabled, using the new repo as cwd/PYTHONPATH. Log/record exclusive creation refuses an existing observation. Receipt records source, target/Python identity, command, exit and log SHA, with CPU-only scope. A timeout can leave a partial log without a final observation; this is incomplete evidence and must not be retried as if nothing ran. No actual target CPU run was executed during review.

`transport.py` retains strict host-key checking, disabled host-key updates, batch/identity-only SSH and existing task-specific public route configuration. It reads configured credentials and does not rewrite them or global SSH settings. Default target remains h100-5. The h100-10 branch is scoped operationally to an independently reviewed read-only payload reading this task's NFS marker. It is a general Python transport and does **not** technically enforce read-only payloads; this review authorizes no other h100-10 code or experiment operation. No SSH command was executed by this reviewer.

`transfer-source.py` hashes source parts and the whole incremental bundle, checks returned fixed080 identity, and transfers each bound tool separately. Its calls use `check=True`; failures stop and preserve request evidence without automatic retry. Final transfer manifest records source/bundle/tool hashes and source-only scope. It does not submit Slurm or call training.

Local syntax compilation of all four Python tools passed without executing their remote CLI entries. CPU/SSH transport were inspected statically; installer/client guards were exercised only with the local fixtures described above, never against a target.

## Final reviewed byte identities

| Artifact | SHA256 |
| --- | --- |
| `deploy.py` | `134f2cb125d6489aeb6cf5b84f124e6c632931d29c3e80c92984d95960669e40` |
| `run-target-cpu.py` | `da02b61135799b84c2debda99ca9d86bef064398d82ecab73139ca4f813767ae` |
| `transport.py` | `06d793daa3abb31e6387cb8c56befde9d38d28d0a07248fd9c629c1efcf8f706` |
| `transfer-source.py` | `223b8b39a6d31bbfc2e95e6d2b43a9a3f51b6b6b1df9a31d9c927e8f40a918bd` |
| `DEPLOYMENT-REVIEW-TESTS.json` | `0af70c5088a4dd1cbed0948deeb2d40cce30981cc12506d474d865d221d06aa7` |

Review attestation: the PASS result applies only to these bytes and the source-only/task-scoped limits above. Only this requested review and local fixture result artifact were written by the reviewer; deployment/training source, credentials and remote state were not changed.
