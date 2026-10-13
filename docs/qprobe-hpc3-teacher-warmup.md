# HPC3 question probe: one frozen-teacher warmup plus 200 training outers

This experiment starts from the official Qwen2.5-3B-Instruct revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`. It reuses read-only HPC3 model,
canonical common data, retrieval assets and the existing Python3.10/Torch2.4/
vLLM0.6.3 runtime. It has its own allocation, source snapshot, services and output.
The existing CoSearch and TOP-D jobs are not deployment targets.

Before the first formal actor update, draw a fixed seeded pool of 5120 unique
canonical training questions. Generate five complete real search trajectories
for each question using the still-frozen official actor. Ten immutable chunks
contain all 25600 trajectories, their unique identities, strict terminal EM
vectors, correct count K, mean and sample standard deviation (ddof=1). Preserve
all K=0 and K=5 groups. Infrastructure/provenance failures cannot become five
zero labels. Cache identity pins exact questions, model files, sampling settings,
scorer source, feature namespace and retrieval version. Full SHA readback is
required before reuse; accepted chunks are never regenerated on restart.

The warmup head bootstrap fits **all 5120 groups**, without the legacy historical
bootstrap cap of512. It uses the existing six-count classification and mean/std
auxiliary objective. Warmup is one additional collection outer, not an actor
optimizer update and not one of the 200 successful training outers. Existing
actor LR warmup57 and all original reward/advantage/PPO guards remain separate.

Actual eight-rank pre/post readbacks must prove unchanged model parameters,
empty/unchanged Adam, scheduler and zero completed actor outers. Restore the
pre-warmup worker and driver RNG boundary so a cache hit does not change the
formal actor's initial sampling state. Candidate selection uses its own stream;
warmup does not consume the normal training sampler. The head is saved at outer0.
Restoration of a trained actor uses its paired checkpoint and does not rerun
teacher warmup or initialize a new head.

After warmup, use the documented question-probe flow: 5120 candidates, E=64
uniform exploration followed by448 highest predicted expected sample std,
512x5 real trajectories, original actor update, then one online head fit for
the next outer. Keep successful training outer budget200 and prohibit update201.

The one-GPU feature preparation entry is
`reproduction/hpc3_qprobe/features.sh BASE IMMUTABLE_SOURCE`. It only prepares
fixed question features; submission or feature completion alone is not proof
that warmup or formal training has run. Runtime acceptance requires actual warmup
25600 trajectories, all5120 fitted labels, eight-rank frozen-state/RNG evidence,
then actual complete formal updates and eventual outer200 acceptance.
