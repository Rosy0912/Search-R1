# Independent hardware runtime preparation review

Reviewed locally on 2026-10-09 by the actor review agent. Scope is `prepare_hardware_runtime.py`, including the author's repairs after independent feedback. No remote execution, deployment, GPU work, training, or production source edits were performed.

**Conclusion: no remaining must-fix finding in the reviewed bytes.** The helper prepares a source-bound interpreter/supervisor anchor, with explicit hardware-only scope. It does not establish hardware acceptance or grant training permission.

Current helper SHA-256: `d4abc3697025788eb53d2ccb72163aed87e1433575f2f03c598fa8b0bb71be79`.

## Verified bindings

The helper requires actual Linux h100-5 and the exact new environment prefix under `/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009`, canonical root, clean repository, and source `080ebbc46b89d61567cd33e1a5e2f1b40afd8b84`. Its `require` checks remain active under optimized Python.

The CPU prerequisite is confined to regular canonical `ROOT/logs/target-linux-cpu-compatible.json` and `.log`. The receipt must have the same source, successful exit, 255-pass tail, actual host/platform/environment/executable identity, exact CPU-only scope, and the complete command from all 23 suites in the SHA-pinned reviewed runner. Both receipt log hash and actual log digest must equal the independently reviewed remote log SHA `7eb44f559e071ef67a1ba035590753c71de0e914aeb03116f978ece716205cb0`. An old60/3315 source or old/foreign receipt cannot satisfy these checks. The initial version's unrestricted receipt log path and self-selected log hash were repaired.

The helper pins installed supervisor bytes to:

| Tool | SHA-256 |
| --- | --- |
| `resource_watch.py` | `5ea7c46d2a8b4cda5e666cacb21b781a1d2a8908cd605950f3fae6c0d1774025` |
| `owned_processes.py` | `2c8cdb3363461b402c3155091cab89739524256b1f89ab58c0f5e62334a877bd` |

It copies those exact bytes to `ROOT/tools`, providing the path required by source080's live guard. It captures actual `/proc/self/exe` resolved path, device/inode and file SHA, and binds the capture and tool manifest into a new `BAPO_ALIGNED_EM200_LAUNCH_V1` anchor. Record, manifest and approval keys/scopes match the rebinding watcher's `load_runtime_anchor` requirements. The helper checks the imported watcher module's actual file is `ROOT/tools/resource_watch.py` before invoking `load_runtime_anchor(..., verify_current=True)`. A cached foreign watcher cannot produce a success observation through this helper.

The CPU suite variable is now separate from the two supervisor names. Independent review caught and the author fixed an intermediate overwrite that would have attempted to copy 23 suite files as supervisor tools; the final admission-block test confirms both supervisor names remain intact.

## Write and partial-state boundaries

Every destination is fixed under the new root. Existing files or symlinks in any of the six destinations cause refusal before preparation. Each write requires a canonical parent, rejects a symlink destination, uses exclusive binary creation, flushes and fsyncs. There is no directory creation, arbitrary supplied output path, overwrite fallback, or GPU/process launch. If a later write or final anchor check fails, earlier writes are retained; a repeat encounters that partial state and refuses rather than silently replacing it. This is deliberately fail-closed preparation, not a multi-file atomic transaction or automatic recovery.

The generated approval explicitly says `HARDWARE_ONLY_RUNTIME_ANCHOR_NOT_TRAINING_AUTHORIZATION`, and its scope says no GPU work, no hardware receipt, and no training authorization. The interpreter record likewise excludes GPU/launch acceptance. Downstream launch authorization still requires its own review and valid human instruction; this preparation must not be cited as that authorization.

## Local checks and limits

Four actual write-helper checks passed: first exclusive write; refusal to overwrite; refusal of symlink parent; refusal of symlink destination, with original and foreign target contents preserved. Twelve exact CPU-admission-block checks passed: valid source/identity/suite binding; rejection of wrong host, source, failed exit, scope, external log path, old environment prefix, changed log hash, wrong actual prefix/executable/platform, and an omitted suite. The test also asserted supervisor-name preservation.

These admission tests used a canonical temporary root and a sentinel CPU log with only its digest mocked to the expected reviewed hash, because the complete actual target log was not supplied locally. The reviewed runner's actual bytes/SHA and real receipt structure were used. Therefore these tests establish branch and binding behavior; they do not claim a local recomputation of the target full log hash or execution of the complete Linux preparation helper. The final actual target invocation must independently produce a successful `verify_current` result and matching persisted anchors.

Signed: independent actor review agent, 2026-10-09.
