# Independent target Linux CPU result review

Reviewed read-only on 2026-10-09 by the actor review agent. No remote operations or source changes were performed.

**Conclusion: the supplied target receipt supports 255 passed, 6 warnings in 51.79 seconds, exit 0, across the complete 23-suite CPU compatibility run on Linux h100-5. No must-fix discrepancy was found between that receipt, the installation receipt, and the previously reviewed harness. This is CPU test acceptance with disclosed fixture adaptations; it establishes no production GPU acceptance.**

## Source and harness binding

The algorithm source is `080ebbc46b89d61567cd33e1a5e2f1b40afd8b84`. The reviewed runner requires that HEAD and a clean repository before starting pytest. Its actual child interpreter identity in the receipt is Linux/h100-5, prefix and executable under `/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009/env`.

All three installed tool files match local bytes and the bytes in tools commit `ca4f26e1679977407b9efc5f4dcd723080ff8885`:

| Installed file | SHA-256 | Bytes |
| --- | --- | --- |
| `aligned_linux_cpu_plugin.py` | `37ffb4e95ea329b0eb1bd8d9e86b474fc23d8ce754fbc27718f553502804bfe4` | 1090 |
| `run-target-cpu-compatible.py` | `7c9bfb09946629faa91581bd9e10c968bd3da7544bf6b9543a195c6ddf37170f` | 3393 |
| `LINUX-CPU-COMPAT-REVIEW.md` | `3215eb98ff1179da4630fbbc1fa5bfd85d96bb6aa9d14ec4eba4e4a434ffe8f8` | 4002 |

The recorded command exactly matches the reviewed runner's target Python, `-B -m pytest -q -p aligned_linux_cpu_plugin`, and all 23 suite paths. No suite was omitted. The remote wrapper runs the installed compatible runner directly. Installation completed before the reported run. Both installation and test transport records have exit 0 and empty stderr, and their parsed stdout matches the corresponding actual JSON receipts.

## Adaptations and practical limits

The receipt explicitly discloses actual-source test namespace binding and the single exact CPU ToyHF oracle's existing naive CE backend, restored by pytest monkeypatch teardown. These adaptations resolve the two earlier environment/fixture failures while keeping source080 production and test bytes unchanged. Prior independent review verified exact-node isolation, neighboring-node exclusion, CUDA-visible refusal, teardown restoration, actual fixture import, unchanged suite selection, and local execution of both formerly failing tests.

The reviewed runner hides CUDA and applies offline and no-bytecode settings only to child processes. Its receipt scope is `ACTUAL_TARGET_LINUX_CPU_WITH_EXPLICIT_TEST_ONLY_COMPATIBILITY_NOT_GPU_TRAINING`. Six warnings are shown: Ray deprecation, expected CPU CUDA-autocast disablement, and existing OmegaConf warnings. The receipt reports no failed or skipped tests. GPU FlashAttention execution, actual distributed model training, runtime retrieval identity, first-five optimizer/stage commits, checkpoint reload, and continuation remain outside this CPU observation.

The compatible run's remote full-log SHA is recorded as `7eb44f559e071ef67a1ba035590753c71de0e914aeb03116f978ece716205cb0`. This review independently checked the supplied receipt and its transported tail; the complete compatible log was not supplied locally, so its full-log hash has not been independently recomputed here. The tail ends with `255 passed, 6 warnings in 51.79s`. UTC start/end are 2026-10-08 19:02:42.131513 and 19:03:35.156688 (2026-10-09 locally); wall time includes runner/process overhead beyond pytest's duration.

The original separate receipt remains exit 1 with `2 failed, 253 passed, 6 warnings in 51.47s`. It was not overwritten or retroactively declared passing. The local transported original raw log has one additional terminal newline: removing only its final byte reproduces that original receipt's full-log SHA `bddc32861628f43fdc559d7e6994ef2fb33440ba41307a79a37afed974795e26`. This transport formatting difference does not change the retained original failed observation.

## Reviewed local receipt hashes

- `aligned-em200-target-linux-cpu-compatible.actual.json`: `dc45869319022f2127b146db707861454da26ee2f9422b2e4fd15d6b6cf54bee`
- `aligned-em200-cpu-compatible-deploy/install.actual.json`: `7153ef14e7c0540168e0d11429cca97fe0139cf6856ea6c672ed7a6fce08a42a`
- `aligned-em200-target-linux-cpu.actual.json`: `324b8ca18de2435282eca8fb51c91982d8725262fc1f8f9aa1411aa1d0529a62`

Signed: independent actor review agent, 2026-10-09.
