# HPC3 TOP-D preparation

Status: runtime validation pending. GPU training has not been started or requested.

Target: `/data/user/jxu957/experiments/topd-20261008`. The verified original `bundle` is immutable. `runtime` is an isolated infrastructure port, with hardlinks to the same verified model/data/index bytes.

The active training protocol remains the user-approved teacher HF step85, official Qwen2.5-3B-Instruct initial actor/reference, common manifest `0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24`, and original TOP-D student300 overrides. The student has completed zero updates. Only `training_overrides` is active; the inherited historical teacher command is provenance, not an executable student command.

FA3 is built from source commit `5231d95fe13733fb534c01895f7ea88c6a6c7793` with CUTLASS `c506e16788cb08416a4a57e11a9067beeee29420`, CUDA12.8 and GCC10.5. The complete adapted setup source is `fa3-setup-hpc3.py`. Independent distribution `flash-attn-3==3.0.0b1` prevents FA2 installation from removing its modules. Existing VERL/vLLM code still needs FA2 interfaces; installing FA3 does not automatically select it for those calls. FA2 builds from retained internal 2.6.3 source with CUDA12.1, matching Torch2.4.0+cu121. The activation preloads CUDA12.8 runtime for FA3 while preserving the matching Torch JIT toolkit and explicit GCC10.5 compiler paths.

Internal pip: `http://harbor.internal.com:8081/repository/pypi-hkust/simple`. No automatic upstream CUDA/wheel fetch or Mac package/model relay is used. Wheels, libraries, environment, caches and actual Ray files remain remote/shared.

GPU FAISS1.8.0.2 is isolated under `environment/gpu-faiss-lib`. `repair_faiss_elf.py` adjusts only invalid load-segment alignment for glibc2.28 and retains the original bytes/hash; it does not modify instructions. Native import is verified; GPU execution remains untested.

The HPC3 controller lacks pidfd. Its replacement serializes Python child creation and reaping, pauses owned parents before identity-checked signalling, and preserves unrelated baseline processes. Actual CPU Slurm tests cover concurrent guardian cleanup/poll/blocking wait, orphan adoption, TERM resistance, unknown ancestry refusal/resume and rejecting new spawn after revocation. The supervisor itself does not run Ray or install a SIGCHLD reaper; Ray runs in its separately owned training child.

Preparation tools build exact manifests and check the unchanged training recipe. Source/student/teacher/data/retrieval fresh SHA checks have passed. Final validation exercises native imports, actual Hydra CLI composition, offline tokenizers, dataset tokenization/collation/DataProto, actual CPU Ray tasks with native modules and observed worker-class serialization, and the full21,015,324-vector index plus corpus matches.

Prepared launcher: `runtime/tools/topd-instruct300-launch-v6/launch.sbatch`. Its template is `outputs/training-preparation/hpc3-config-draft.json`. On a separately authorized GPU launch, it binds the actual Slurm node and a unique job output directory before starting the original supervisor; it requests one node/96CPU/768GiB/8GPU. Preparation scripts do not submit that launcher.
