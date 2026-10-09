# Independent teacher external-wait audit

Local CPU candidate; root must publish and read back the exact sources before actual data execution. Production training/evaluation/backup code is unchanged. This CLI reads only selected small teacher JSON and historical phase-exit JSON from explicit accepted backup inventories. It imports no Torch/model/tokenizer, performs no Node/HTTP/process/RNG action, and is never a GPU dependency.

The pinned `aggregate_clipmass.py` (SHA `b7a264119fd18b9ac744f452c86e8deb6db550c318050178e94adb38a1f39c8a`) is compiled from fresh SHA-checked bytes in a separate namespace; no pyc. Its `FixedBackups` handles accepted direct/gzip inventory, selected member SHA, path/stat continuity and gzip CRC. Gzip remains sequential: it decompresses skipped archive bytes but does not parse/hash token or NPY payloads. C1's small closed-global archive can avoid traversing its large raw archives for phase-exits only, while parent/selection/child/request remain in accepted outer archives.

## Input

Use one immutable JSON, with exact local path/SHA refs:

```json
{
  "schema": "opsd-ablation-teacher-terminal-audit-inputs-v1",
  "parser_source": {"path": "/ABS/published/aggregate_clipmass.py", "sha256": "b7a264119fd18b9ac744f452c86e8deb6db550c318050178e94adb38a1f39c8a"},
  "producer_source": {"path": "/ABS/frozen871/controller.py", "sha256": "c8aa9c8f81bb669707085d0a586843608058258a3882006b380a28c556f26b56"},
  "facts": {"path": "/ABS/accepted40facts.json", "sha256": "ACTUAL"},
  "groups": [{
    "variant": "C1",
    "config": {"path": "/ABS/config-local-original-bytes.json", "sha256": "ACTUAL"},
    "source_config": {"path": "/data/EXACT/config.json", "sha256": "ACTUAL", "bytes": 0},
    "backups": [{"kind": "gzip", "inventory": {"path": "/ABS/inventory.json", "sha256": "ACTUAL"}, "complete": {"path": "/ABS/accepted-or-reader-adapter.json", "sha256": "ACTUAL"}}],
    "external_readbacks": []
  }]
}
```

`source_config` must be the **full original fileproof object** from the accepted facts `configs` (all original metadata fields, not this illustrative minimal JSON). Local config bytes/SHA/size must match; original teacher request's full config proof must equal it. Each explicit group requires all ten accepted native facts with the same run. C0/C1/M/R or later W/P/J use the same interface. No fallback to mutable latest or stage-selected filenames.

Reuse actual clipmass `backups` specs for the group's parent records. C1 adds its accepted 151-member closed-global gzip (39 phase-exits). M/R accepted generic complete proofs require the existing published reader adapter if the three original `FixedBackups` flags are absent. No original receipt is rewritten. Duplicate source members across scopes are rejected; use disjoint outer/global backup scopes.

For C0 the accepted direct outer archives lack external `runs/phase-exits` files. First run outputs exact `missing_external_read_requests` from parent `exit_proof.actual_terminal`, with original path/SHA/bytes. Root may obtain only these small immutable bytes, then use a **new manifest/output** with:

```json
{"source": {"path": "/data/EXACT/phase-exits/original.json", "sha256": "ORIGINAL", "bytes": 123}, "local": {"path": "/ABS/fresh-small-original.json", "sha256": "ORIGINAL"}}
```

Unused, duplicate, altered or wrong-group supplements reject. This supports exact content identity readback, not fresh remote process/stat attestation.

## Invocation and scope

```text
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python3 -B audit.py \
  --manifest /ABS/input.json --manifest-sha256 ACTUAL_SHA --output /ABS/NEW/private-result-dir
```

New directory0700, no overwrite; errors retain `failed.private.json` and exit nonzero. A successful read can still produce `complete:false` and exact missing requests. Do not call that full acceptance.

Necessary original871 `Controller.verify_exit` semantics (lines305–318) are implemented explicitly: same run/config/phase and exact child-exit evidence; actual wait0, UID/PID/starttick/run, no uncertain identity; parent embedded `gpu_release.all_owned_release.accepted`. Extra actual schema/variant checks are included. Fresh producer SHA binds the reviewed source, without importing its Controller or calling live gates. Teacher child/request/selection/source100, count<=8, common BF16/AR/no-denominator flags are bound. NA slots require null child/request/exit and count0.

**External wait bytes are independently read. Release is historical parent-embedded evidence only.** Original `actual_terminal` file persists child wait codes before release; it does not contain a separate release file. This helper never claims fresh independent release execution, live process absence, case/HTTP/token quality, all70 debug coverage, final24GiB ledger, or scientific effect.

One tiny necessary negative example rejects a differing external child's starttick against parent evidence. It is not a production/archive acceptance result. No broad suite or actual data run has been performed for this candidate.

## V2 narrow compatibility correction

V1 actual local run PID59039 exited1 before terminal acceptance: original871 teacher request.config omits `mode`, while accepted facts fileproof adds `mode`. All seven original producer identity fields matched. V1 failure and sources remain preserved. V2 only changes this cross-producer comparison: path/SHA/bytes/device/inode/mtime_ns/ctime_ns are required and exact; mode must also match if request supplies it; unknown/missing request identity fields reject. Full facts source_config remains intact. Other dict equalities compare same actual original request/child/parent bytes and are unchanged. Input schema/CLI are unchanged, output schema is v2. No actual data execution or repeated test has occurred for V2; original tiny negative example is retained, not rerun.
