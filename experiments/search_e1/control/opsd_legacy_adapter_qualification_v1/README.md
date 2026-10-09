# Optional historical L adapter-only CPU qualification

This separate stdout-only entry implements the already approved minimal design. It does not change the active seven-group driver, any training source, model or 5632-slot evaluation. Root publishes and reads back this source before one actual Node CPU invocation. No real qualification has been performed by preparing this entry.

The unchanged published reader is SHA-bound and source-byte compiled. Its `Reader`, `bind_dependencies`, `checkpoint_identity`, `load_adapter`, and `adapter_delta` are reused. Only the original small helper/config/runtime/source manifest/cohort qualification files and the two three-file adapter directories are content-read. Source100's twelve expected file identities receive metadata checks, with no model-weight content hashing. Rank-state canonical paths/declarations are checked by `checkpoint_identity`, but no rank-state/Adam/RNG payload is read. There is no model, optimizer or tokenizer construction, RNG restore, generation, HTTP request, signal, lock or Node output write.

The target is the actual old forward checkpoint99/outer9, not a new reverse group: original run `1d1d67b60f0e46a2afc0536a36ab69b1`, original config d0c30… and target manifest476c1…. The explicit baseline is old checkpoint0/outer0 manifest b8ed9… from the accepted original protected-state inventory; common-initial manifest641e… binds its adapter. Each adapter is fresh SHA-read using the reader's opened-file identity checks, loaded to CPU, required to have exactly504 finite FP32 tensors and the exact declared tensor digest. Common initial must have252 zero B tensors. Actual target differences are computed against that exact common adapter.

Use one immutable JSON with schema `optional-L-adapter-only-inputs-v1` and `{path,sha256,bytes?}` refs named `reader`, `config`, `target`, `baseline`, `common_initial`, `common_cohort`, `qualification`, `qualification_source`. The first six historical artifact paths/hashes are pinned in `PINS`; the existing compact qualification and raw-source proof contents are pinned too. Root may copy those two private small proofs to a Node receipt location and supply the copied path while retaining their exact SHA. `reader` points to the published read_checkpoint_cpu.py exact7b260… bytes. No private question/token body belongs in Git or stdout.

```text
env -u E1_RUN_ID CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  /ABS/old/env/bin/python -B /ABS/published/qualify.py \
  --manifest /ABS/immutable-input.json --manifest-sha256 ACTUAL_SHA
```

The stdout `accepted:true` will qualify saved adapter content only. It will not accept full Adam/eight-rank numerical restore, native512 performance, matched effect, or the complete study. Missing, changed or incompatible input rejects before acceptance; original files remain unchanged. Historical ten-outer cohort equality is retained through the accepted SHA-bound prior proof rather than rerunning mining or scoring.

Two stdlib CPU negative fixtures cover a wrong common checkpoint before any tensor access and one non-FP32 tensor. They do not access Node, real adapters or CUDA:

```text
python3 -B -m unittest discover -s experiments/search_e1/control/opsd_legacy_adapter_qualification_v1 -p 'test_*.py' -v
```
