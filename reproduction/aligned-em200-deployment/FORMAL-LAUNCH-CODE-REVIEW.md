# Formal launch independent code review

Decision: **PASS_CODE_ONLY, no remaining training must-fix in the reviewed snapshot.** This document does not authorize submission, certify target GPU execution, accept hardware job3345, or sign a training completion/release receipt.

Reviewer: alignment_generation_review, independent of the formal launcher author. Review performed locally on 2026-10-09. No production source was changed; no remote commands, scheduler submissions, Ray heads, retrievers, or training processes were started.

## Scope and findings

Reviewed the formal Slurm/worker/inner/driver/admission/submission chain, the process-local Ray CLI compatibility shim, and the new formal service ownership proof against frozen algorithm source `080ebbc46b89d61567cd33e1a5e2f1b40afd8b84`.

The worker validates the actual Linux interpreter/source/tool/runtime bindings, fixed common169028/data/order manifests, unique submission receipt/token/job, and independently accepted hardware evidence before claiming a canonical fresh RUN. The source080 exclusive startup claim occurs once before cleanup, supervisor logs, services, or training output. Durable submission intent prevents an unknown outcome from being retried. Admission remains deliberately unbound: hardware/authorization SHA pins are None and the formal file map is empty. These candidate bytes cannot contact Slurm or create a fresh claim until root binds the real accepted originals and republishes the resulting bytes.

Resolved review blockers:

1. The former retriever PID/environment-only readiness proof now freezes the actual source080 CUDA/GPU-FAISS command, parent PID, UID, executable inode and process start ticks before readiness. The identical birth is checked after readiness, after source retriever admission, and before the driver enters main. The reused birth helper is included in the formal source pin set.
2. The former Ray TCP/CLI-alive check now requires the actual sole GCS listener owner, same RUN/job/UID and stable parent chain to the exact launched Ray CLI. Driver verification connects only after this proof, checks one live loopback node with eight GPU resources, its own raylet listener/CLI ancestry and exact short session directory, shuts down the temporary verification connection, and checks the frozen service births again before exec. Foreign ancestry and listener swaps fail closed.
3. The unconditional newer-Click Sentinel import failed the real supplied local Click8.1.8/Ray2.40 help entry. Feature detection now supports older Click and retains explicit copy/deepcopy identity checks when Sentinel exists. The change is confined to the Ray launcher process and does not edit dependency files.

The wrapper exports the official initial Qwen model, pinned common169028/prepared data/observed order, approved archive golden and own loopback URL before Ray head creation. It preserves the source080 production profile: four search interactions plus final no-search generation, HF frozen denominator semantics, production stage commit and training EM, and same-process first-five pause/release. The final exec enters the existing preflight and `search_r1.baselines.main` directly with the production config; it does not invoke source run.sh a second time, resume a prior run, bypass the gate, or alter actor mathematics. Source080 main performs its normal config/asset/retriever validation and Ray main_task dispatch. Wrapper driver exec does not reset or replace the source trainer process at its first-five gate.

## Actual local verification

From the tools checkout, the supplied dependency interpreter ran:

- `python -m pytest -q reproduction/aligned-em200-deployment/tests/test_formal_launch.py`: **8 passed in 0.03s**. Tests cover the unbound no-scheduler boundary, old/submission receipt rejection, durable unknown-outcome nonretry, actual source startup claim exclusivity, receipt identity mismatches, Ray ancestry/port-swap rejection, both Click feature branches, and wrapper claim/entry ordering.
- `python -B reproduction/aligned-em200-deployment/ray_cli_compat.py --help`: **exit0**, actual local Click8.1.8 and Ray2.40.0. This only loaded the CLI/help; it did not execute start. One existing urllib3/LibreSSL compatibility warning was printed.
- `bash -n` for formal.sbatch, formal_worker.sh and formal_inner.sh: **exit0**.

Process ownership tests use local boundary fixtures; they are not actual Linux /proc/GPU acceptance. Newer Click Sentinel behavior is a synthetic enum branch test, not an executed target dependency claim. The actual node's Ray CLI/GCS/raylet/session and CUDA/E5/GPU-FAISS compatibility must still satisfy these guards during a separately authorized target execution. Hardware3345 was last known pending from saved evidence; no terminal or hardware success is inferred here. Formal admission binding changes require fresh exact-byte review/publication. This report cannot substitute for the independent actual hardware receipt, formal authorization, first-five release, or EM200 completion review.

## Frozen bytes

| File | SHA256 |
| --- | --- |
| `formal.sbatch` | `1ac3e2a317488593d18f0fdecb130c068a30e8987540cbd67f8739980c9f81cd` |
| `formal_worker.sh` | `f65e3d8cde4aa935d0fe941da11840c003dc6d381e81901f59a4c622cd7efcb7` |
| `formal_inner.sh` | `671625f8b8671518144a09feba280a58dceb303a1eccc399a68810b00f263cbc` |
| `formal_driver.py` | `6c32119ec6e0267c869f9fb035ce584f6cc867bab88583b4e20fd3e2e14c943a` |
| `formal_service_proof.py` | `704909275113fe079c287e8652c07ef000e4cfa06f7cf904781b93dc9bed6a96` |
| `formal_admission.py` | `4bdb78376158c3a22a549cabbfdc05b594684bb9a8214611c0df3aa8281d671f` |
| `submit_formal.py` | `d2800301767a62ef41e65da5baeed38ad58a69505ce655cc81cb9d5b57a70562` |
| `ray_cli_compat.py` | `a6ea15b2ac65ff347ca574ed3af9688c22777723d6a2ddbd7b88f33455f6ddc3` |
| `bind_retrieval_proof.py` | `3ae4f0095b775716b0f2d3180b513906a7be51837d1f2786b3ec72447733704d` |
| `tests/test_formal_launch.py` | `a7bb2a6b85868788b3912a731ec98d177dd93eb71686627df4f18476ab123b43` |
