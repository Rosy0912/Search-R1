# Independent count-backend and E1 integration review

2026-10-04; CPU/static only. Candidate fingerprints are in `count-integration-independent-files.json`. The four trainer/gate files remain frozen at `trainer-gate-files.json`; this review does not alter state/profile/production source or launch jobs.

## Count backend: CPU review passes; actual NCCL pending

`distillation.global_token_count` still sums the raw mask once and sums that scalar across the default distributed group. The new branch puts only this one-element FP32 scalar on the rank-current CUDA device for NCCL; for Gloo it is CPU-resident. World size, denominator, mask and gradient mathematics are unchanged. This addresses a real supported-path error: DataProto masks can be CPU tensors, while NCCL does not accept CPU all_reduce tensors.

Independent CPU run: `test_opsd_count_backend.py` plus `test_opsd_unclipped.py`: **6 passed in6.41s**, stdout `count-backend-independent-cpu.txt`. Actual two-process Gloo verified unequal local counts3+1 -> global4 on each rank and globalempty0 without skipping a collective. NCCL routing was explicitly mocked and verified scalar-only/current-device7 routing, world8 and unchanged input mask. This does not prove actual eight-rank NCCL or native GPU objective equivalence. The NCCL branch still needs the owned native first-five/device gate. Backend names `nccl`/`gloo` match normal default groups; other/custom mixed backends retain the previous route and are outside the E1 supported topology.

The change preserves token-mean scaling used by the actor; no PPO/reference KL, clipping or optimizer-clock change is introduced.

## Source closure: not yet accepted as a complete downstream proof

Driver gate's frozen implementation verifies seven critical modules and selected actual live functions against source code structure. Its real seven-module CPU import test passes. That proves those critical objects; it does not check every other imported project module/function, nor does a driver check automatically prove downstream Ray actors' executable imports.

The profile bootstrap's `verify_imports` pins every already-imported project module's actual source path and bytes. This is the appropriate stronger closure to reuse downstream. However, the currently inspected `check_live_code` silently skips a function whose code filename is outside the `allowed` map or synthetic. Concrete CPU reproduction: a `search_r1.baselines.synthetic_audit` module with `def patched(): return "UNREVIEWED"` compiled as `<unreviewed>` returns0 from `check_live_code(m,{})` instead of rejecting. In a real module other valid functions can keep the aggregate count positive. A live project patch can therefore escape the current full-function code proof. Class `property` getters/setters are also not traversed.

This is a **blocking gap for claims of complete live-code closure**, not evidence of an existing production mutation. It was sent to the profile author for a minimal strict-origin fix and meaningful synthetic/property rejection tests. Generated dataclass/library functions need explicit, narrowly described exceptions; project-owned executable functions cannot be silently exempted merely because their filename is unexpected.

Worker full-state `recovery_source_closure` currently records actual loaded project module paths/bytes, exact worker path and commit. This is useful identity evidence but does not independently compare all live functions against compiled reviewed bytes. Its contents can legitimately be process-specific; driver and Ray rank module inventories need explicit reviewed expected coverage, not merely an assumption that an identical Git commit proves actual imports.

Required integration follow-up: profile checker strict all-project live-code origins; driver verifies all loaded project modules in the externally pinned closure; each Ray worker's own state evidence independently binds its actual source closure and live executable implementation. Root requested this delta; four frozen trainer/gate files have not been changed during this review. No native GPU closure/recovery receipt is approved here.

## Closure follow-up

The reported synthetic/unknown-path/property gap was corrected by the profile author, with project aliases also subject to strict origin checks and known decorator-library byte/code pins. Actual full67-module driver+seven-critical integration now passes, and actual CPU invocation of the worker-local checker reports67 modules/1069 functions. Combined97 CPU tests pass; details and precise SHA freezes are in `TRAINER_GATE_IMPLEMENTATION.md`. The original failed collect caused by noncanonical frozenset constants remains archived separately. This closes the reviewed CPU/static integration gap, while real Ray GPU ranks still must supply their own independent actual proof before first-five acceptance.
