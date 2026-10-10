# Job3360 final HF/tokenizer cold reload acceptance plan (preparation only)

Training3360 remains untouched. No weight load, GPU request, CPU job submission or remote mutation occurs during this preparation. Existing exhaustive CPU verifier is already present in the frozen8b6 source package:

`/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/branpo-fresh200-8b6f99d1b3353145200eb3fa7c16a579a79a987c/source/repo/reproduction/baselines/tree-validation-20261002/evidence/runtime/check_real_checkpoint.py`

SHA256 `b3075d7b9346bb7682e0ea05962de0624c117b354590b728ed8a18b7e8ba3c97`. It performs actual CPU FP32 AutoModelForCausalLM.from_pretrained, rejects every missing/unexpected/mismatched key/error, traverses every official-base safetensor parameter and validates full shapes/finiteness and nonzero deltas, permits only genuinely tied extra state keys, actually reloads tokenizer and checks vocabulary/BOS/EOS, performs one short CPU forward, and hashes every HF file. It does not use8192 sampling.

It is applicable on h100-10 with existing tree-grpo/env Python and offline assets. No GPU is needed; use CUDA_VISIBLE_DEVICES='', clean PYTHONPATH and bounded eight CPU threads after trainer3360 has terminal exit0 and checkpoint200/native manifest fully published. Cold FP32 model needs roughly12GB parameter RAM plus shard/delta/forward overhead; use an available64GB CPU memory budget, no concurrent model loading while3360 is active. Root chooses existing authorized same-node CPU execution boundary after final training, without changing GPU reservations/foreign jobs.

Old verifier report alone lacks native3360 source/config/manifest binding, pre/post immutable file checks, and PAD/fullspecial/chat template identity. A minimal separate wrapper will add those checks while calling the unchanged frozen exhaustive verifier. It lives outside frozen runtime and requires independent context-free source review before actual execution.

Bind source `8b6f99d1b3353145200eb3fa7c16a579a79a987c`, config `c9415c0096853f78d870e205fc956f20045cde985836de5225f609eaf091d4b3`, run `tree-branpo-fresh200-3360`, package manifest `e2503977c8746edbc83b722d8224548f060909df9fc25d91a9cb88a947a79a21`.

Inputs remain remote: officialaa8e725 base, run/checkpoints/actor/global_step_200 HF directory and run/checkpoints/training_state/global_step_200/manifest.json. Outputs remain remote under run/output/final-hf-acceptance: full-HF-cold-reload.json, tokenizer-cold-reload.json, identity-bound-final-hf.json and verifier.log. Only the small identity-bound-final-hf receipt is returned locally after completion; no weights, tensors, logits or optimizer state transferred.

## Exact execution entry (after independent context-free review and terminal3360 exit0)

Standalone wrapper file: `verify_final_hf_3360.py`, SHA256 `f74282095f406c5c7dc9941dc7d61207cc30c0eae5ed2e0968ce88abdba7d70c`. Only this small reviewed code file may be copied through the existing management route into `/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/final-acceptance-tools-3360/verify_final_hf_3360.py`. Do not add it to or alter the frozen candidate package. Record code publication commit and independent review SHA in the external final audit.

Remote CPU command, invoked through the existing `ssh_retained18.route(command)` after the file's actual deployed SHA is verified:

```sh
env -u TREE_INSTRUCT_TOOLS -u TREE_FIXED_ACTOR_PROFILE \
 CUDA_VISIBLE_DEVICES='' PYTHONPATH='' PYTHONNOUSERSITE=1 PYTHONOPTIMIZE=0 \
 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
 /dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python \
 /dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/final-acceptance-tools-3360/verify_final_hf_3360.py
```

Use existing same-node authorized CPU execution boundary after terminal training, without a GPU allocation. Do not run this command while3360 is training. Wrapper checks terminal exact200exit0, actual config SHA, native checkpoint200 lineage, frozen verifier SHA and package SHA; the complete HF inventory must match native publication before/after; the immutable official base must match frozen package identity before/after. Existing acceptance directory causes refusal, preserving failed evidence; no automatic retry.

Wrapper outputs only a small final bound JSON to stdout. Save that receipt to local `outputs/tree-branpo/FINAL-HF-3360.json` only after successful actual completion and two real verifier progress observations (log/file byte advance and terminal return). Detailed full-HF and tokenizer reports remain under remote run/output/final-hf-acceptance. No model parameter, optimizer tensor or logit data crosses the management route.

Static preparation checks passed: AST; ordinary file/full directory identity helper; exact JSON hash pin; refused overwrite/hash drift. These are small synthetic metadata checks, not claims of actual model reload. Final HF acceptance has not executed. Root must perform independent source review, deploy-code SHA verification, terminal prerequisites, actual CPU execution and final result review.
