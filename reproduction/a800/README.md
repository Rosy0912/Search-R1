# A800 weight continuation

The H20 job completed 149 updates and failed with OOM during update150.
Its latest complete checkpoint is100. The user explicitly accepted restarting
from these weights with fresh Adam; RNG and sampler state are not restored.
The49 unsaved updates are lost. This is not state-exact resume.

The original Qwen2.5-3B remains the KL reference and tokenizer source.
Only the actor loads checkpoint100. The first new update is101, using the
original warmup schedule position100/142. Logical updates101–500 comprise
400 actual updates in this segment; threshold501 preserves upstream counting.

Training uses physical GPUs0–3; local FP16 sharded FAISS uses5–6. GPU4 and7
are excluded. The effective batch512, minibatch256, five GRPO responses,
four action budget, generated-answer EM, top3 retrieval, observation masking
and KL coefficient0.001 remain unchanged. vLLM reserves0.4 of each80GB GPU;
this changes available cache memory, not the objective or sampling recipe.
The private Ray instance uses address=local and project-local temporary files.
No global Ray shutdown or foreign process termination is used.

`run_continuation.sh` holds the project job lock, waits for the atomic transferred
checkpoint, checks its complete SHA256, canonical keys/shapes and all finite
tensors, verifies tokenizer identity, creates an exclusive run-ID log and
manifest, and starts retrieval on127.0.0.1:18000. It checks a real HTTP query
before launching training. Root-level retrieval wrapper deployment is checked
against the project file. GPU admission checks use physical UUIDs.

At105, `acceptance_watch.py` verifies five distinct real update metrics,
source/reference file hashes, all fresh checkpoint tensors and actual parameter
change from100. It writes `a800-five-resumed-updates-acceptance.json` while the
same trainer continues. Saving105 alone is not acceptance.

At500, `verify_full_continuation.py` requires ordered real updates101–500,
the matching five-step acceptance, fresh complete finite500 weights and exact
seven-source evaluation counts51713, including NQ3610. Intermediate validation
remains the fixed diverse64-question sample. Entropy/KL are not perplexity.

Independent runtime review found no launch blocker after the deployment,
processed-data, identity and provenance checks were added. Actual pinned A800
CPU regression passed75 tests. Four-GPU NCCL/BF16, two-GPU real-index retrieval
and real3B vLLM generation passed separately; none counts as training acceptance.
The retrieved14-query top3 overlap is41/42, maximum score delta2.18e-5.
The512-query timing uses repeated vectors and excludes encoding/document loading.

Disk source hashes recorded before launch are not proof of loaded module identity.
Credentials, private checkpoint data and complete raw training logs are excluded
from Git publication. Live state must be read from the target supervisor, segment
log and acceptance files; do not restart the H20 job or duplicate the A800 trainer.
