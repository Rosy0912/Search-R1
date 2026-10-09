# One offline required-check supplement

Publish/read back this independent source before executing once in the existing Node CPU environment, with CUDA_VISIBLE_DEVICES explicitly empty. It does not run training, models from source100, HTTP, GPU, Node resource checks, or an unrelated fixture suite; it writes only JSON stdout. Root saves exact argv, stdout/stderr and actual return code independently.

```text
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /EXISTING/ENV/python -B check.py --source-root /EXACT/source/training-871dc46/opsd_ablation_v1 --source-manifest /EXACT/source/training-871dc46/opsd_ablation_v1/source-manifest.json --publication /EXACT/original-publication-871dc46.json --publication-sha256 ACTUAL_PUBLICATION_SHA
```

Original manifest/source hashes and commit871dc46 are fixed in the entry. Publication must be the original accepted file-by-file readback receipt (`remote_readback_sha` and rows `path/equal/sha256`), not a scoped admission receipt with another schema. Modules compile fresh SHA-checked bytes without pyc; the trainer is AST-read only, never imported/main-executed. It extracts the actual rank_batch/batches and checks the actual loss/actual scaled-backward expression. Published math/causal/scaler checks are invoked by explicit method names, not unittest discovery.

For B=1..32, ordinary and mixture targets, heterogeneous masks and positive unequal weights, actual pair_objective/SharedTargets runs on the already published small Toy model. Eight serial replicas—including empty ranks—use actual rank partition and one actual-B divisor. Their loss and preclip SUM gradient are compared with one identical model processing all B trajectories. Tolerances: absolute2e-6/relative2e-5. K0, endcursor and illegal midbatch cursor behavior are checked. This closes the small CPU tail/reverse arithmetic coverage; it is not NCCL/GPU acceptance or a new scientific result.

The fallback is deliberate: no actual nested checkpoint restore or small actual PEFT Adam/RNG roundtrip is implemented. Reused existing scaler-only restore is identified as such; complete production current/outer_old/Adam/scaler/RNG/cursor controlled restoration evidence remains missing. No Toy state reload is labelled full resume. The actual production checkpoints/readers and successful same-run continuation remain separate evidence. This check must never block the seven-group GPU queue.

Preparation was static source review only. No check.py or fixtures executed before publication. A failed check must preserve stdout/stderr/rc; no automatic fallback, science change or retry loop is built in.
