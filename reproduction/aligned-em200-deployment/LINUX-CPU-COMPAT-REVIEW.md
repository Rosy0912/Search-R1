# Independent Linux CPU compatibility harness review

Reviewed on 2026-10-09 by the actor review agent, independently of the harness author. Scope is only the two new test harness files below. No remote execution, production changes, commits, training operations, or GPU acceptance are included.

| File | SHA-256 |
| --- | --- |
| `aligned_linux_cpu_plugin.py` | `37ffb4e95ea329b0eb1bd8d9e86b474fc23d8ce754fbc27718f553502804bfe4` |
| `run-target-cpu-compatible.py` | `7c9bfb09946629faa91581bd9e10c968bd3da7544bf6b9543a195c6ddf37170f` |

**Conclusion: no must-fix findings within this diff.** This is a disclosed test-only compatibility run, suitable for independently rechecking the unchanged source080 tests on the target CPU environment. The original 253-pass/2-fail observation remains a separate observation; this review does not replace it or assert a target compatible-run result.

## Verified boundaries

The plugin changes CE backend availability only for the exact node ID `tests/baselines/test_raw_sampler_diagnostic.py::test_real_actor_forward_recomputes_raw_selected_token_probabilities`. It asserts CUDA is unavailable before making that change. Pytest's function-scoped `monkeypatch` restores the original flag after the test, including normal teardown on failure. The existing naive CE implementation is selected; the real actor forward, input contexts, expected numerical values, and source fixture remain unchanged. No test is skipped, xfailed, or changed to accept different values. Nearby or parametrized node IDs do not match this override.

The test namespace override points `tests` and `tests.baselines` at the actual fixed target repository directories. It bypasses an unrelated installed regular package while importing the original source fixture. It replaces no production package. This hook runs before test collection in the explicit pytest invocation.

The runner checks Linux/h100-5, clean repository HEAD `080ebbc46b89d61567cd33e1a5e2f1b40afd8b84`, and the actual child interpreter's prefix/host/platform. It runs the same 23 suite paths as the original launcher, using the target's own environment, hidden CUDA, offline settings, and no bytecode writes. Plugin activation is explicit through pytest `-p`; production training does not import it. Environment changes are child-process-local. Compatible log and receipt names differ from the original names; exclusive creation prevents replacing earlier observations. Nonzero pytest status is recorded and propagated. A timeout leaves its partial log and does not manufacture a success receipt.

The receipt identifies both fixture adaptations and explicitly limits its scope to target Linux CPU testing, excluding GPU training. Its log SHA binds the test output; the reviewed harness bytes are bound by the hashes above and the deployment tools installation/publication record. The CPU receipt itself does not contain separate harness hashes, so a reader must retain that accompanying tools record for byte-level harness attribution.

## Local checks performed

Six explicit checks passed: exact-node CE override; automatic restoration; neighboring node excluded; CUDA-visible invocation refused before modification; original fixture import despite a competing installed-test namespace; and identical 23-suite selection compared with the original launcher. Both original failing source tests then passed locally (`2 passed`, one expected CUDA-autocast warning). The plugin's repository path was redirected only in this local review process to the same clean source080 checkout; shipped bytes were not modified.

The algorithm repository was verified at source080 with clean status before this review. These results establish fixture isolation and original-test execution locally. Actual target Linux completion, the full 23-suite compatible result, installed FlashAttention behavior, and production GPU execution require their own receipts and are not inferred here.

Signed: independent actor review agent, 2026-10-09.
