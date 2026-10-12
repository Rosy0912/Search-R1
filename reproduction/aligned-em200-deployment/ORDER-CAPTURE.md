# CPU baseline order capture

`capture_baseline_order.py` captures a **new baseline reference on the shared 169028-question training cohort**. It does not claim equality to the old 169103-question experiment or reconstruct that experiment's model-initialized RNG timeline. No model forward, training, generation, GPU allocation, or remote operation is performed.

The bundled `reference/` contains the archive's exact `_create_dataloader` AST source segment, real `RLHFDataset`/`collate_fn`, and original tokenization helpers. `provenance.json` binds the original trainer file and every snapshot SHA. Runtime `verl.utils.model` and `verl.utils.fs` must match the archive hashes. The only dataset subclass addition observes each successful `__getitem__` positional index; it returns the original row unchanged. Every real collated batch's `index` values must match those observed positions and original IDs. The archived DataLoader itself remains unchanged, including native shuffle, iterator seed draw, drop-last, collate and zero-worker default.

The archive currently leaves its prompt-filter expression commented out. This tool preserves that behavior and the actual `truncation='error'` tokenizer path; an oversized prompt fails capture rather than silently dropping a row. The loaded dataframe must remain the exact common training ID sequence. Full common-manifest validation checks the original manifest SHA `0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24`, its prepared/raw files, and common loader identities before observation.

Example CPU invocation on the deployment host, with existing source/model/common inputs and a **new NFS output file**:

```bash
PYTHONPATH="$SOURCE080" "$PYTHON" "$TOOLS/capture_baseline_order.py" \
  --aligned-root "$SOURCE080" \
  --common-manifest "$COMMON/manifest.json" \
  --model "$OFFICIAL_MODEL" \
  --output "$NEWROOT/inputs/baseline-order-common169028.json"
```

The tokenizer loads offline, with remote-code execution disabled. `$TOOLS` denotes this deployment-tools directory, not production source. Output is published with an exclusive atomic link and made read-only; an existing filename is never overwritten.

By default, the tool explicitly sets seed42 at the new CPU reference's **actual pre-iterator boundary** and records the full CPU RNG bytes and hash. Its RNG policy label says this is not historical model RNG. `--preiterator-rng rng.json` instead accepts explicit uint8 bytes from another reference's pre-iterator boundary and binds that file SHA; the resulting order is still observed by executing the real loader, not derived only from a seed.

The output is compatible with production `baseline_alignment_data.verify_order_manifest` and `RecordedOrderSampler`: schema `baseline-aligned-order-v1`, original ordered `question_ids`, one complete epoch of positional `epochs`, batch512, drop-last, zero workers and canonical `order_sha256`. It adds actual batch indices and original IDs, the pre-iterator RNG, archive source hashes, tokenizer/config identities, common manifest binding and tool SHA. There must be **330 actual batches / 168960 emitted questions / 68 dropped questions**. This covers outer1..200 without pretending that a partial200-batch capture is a complete epoch.

Local verification:

```bash
PYTHONPATH=../bapo-baseline-alignment-20261009 ../fresh200-local-test-env/bin/python \
  -m pytest -q reproduction/aligned-em200-deployment/test_capture_baseline_order.py
```

Five tests passed, including real archive dataset and collation on synthetic169028 rows through all330 batches, consumer manifest verification, boundary RNG replay/drift, actual truncation failure, snapshot tampering and exclusive output. The tokenizer is a small CPU fixture. Actual pinned common-data and official tokenizer assets are not present locally; no actual deployment order artifact is claimed by those fixture results. Deployment must execute the command against the pinned existing assets before training admission. A mismatch in archive runtime dependencies, missing original raw/prepared files, long prompt, changed IDs, wrong manifest or existing output is a hard failure that should be reported rather than bypassed.
