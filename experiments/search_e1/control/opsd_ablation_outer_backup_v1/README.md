# Seven-group outer-end backups

This is an independent, unpublished CPU candidate. It copies the reviewed read-only USTAR transport from `opsd_ablation_backup_v1` and does not import or change that helper, the trainer, any model, or any Node asset. Root publishes and verifies the new source before use. No live transfer, numerical Adam audit, training completion or benchmark acceptance is claimed by these fixtures.

The author-approved design is a separate Mac serial queue. It observes only accepted native `outer-complete.json` files for the seven exact C0/C1/M/R/W/P/J runs and outers0–9. Each job permanently pins config/run/variant, actual update count, committed checkpoint path/SHA and completion path/SHA. It never resolves mutable `latest`, accepts pending directories, invents K0 updates, or substitutes a newer checkpoint for a queued job. K0 is backed up if the producer actually created an accepted checkpoint for that outer with unchanged update count. A completion pointing to another outer is HOLD rather than relabeled.

Every package contains the original **19 core files**: checkpoint manifest, one `optimizer-shared.pt`, eight distinct rank RNG/scaler/cursor files, three PEFT adapter files, four training inputs, original config and run identity. Eight declared Adam digests/scaler states and the shared-storage contract are bound, but this stdlib reader does not inspect Torch tensor values. Original shared Adam and rank state bytes are preserved unchanged.

The immutable reference closure is included in the same package: actual common-initial manifest plus every declared payload, exact outer-start manifest and its adapter payloads (prior group checkpoint for outer>0, common initial for outer0), three scientific identity manifest JSON files, native outer completion and goal JSON. Shared source paths at outer0 are deduplicated by exact path/SHA/stat. Original path-bearing JSON files are preserved byte for byte; archive prefixes `group/` and `closure/` provide relocation metadata, not a rewritten runnable config. Source100's twelve model files and the full305 runtime tree are **external assets**, not copied by every checkpoint job; Root must retain their separately verified durable study copies. Retry capsules/debug/raw trajectories remain on Node; a completed accepted checkpoint already records its actual RNG/scaler. Uncommitted overflow state is not presented as committed Adam.

All source files are checked through fixed handles, full SHA and stable metadata; each path component must be non-symlink and regular. USTAR members are exact, deduplicated and reject escape/links/special/GNU/PAX/extra/trailing data. Extraction uses a new0700 private root and performs a second complete local disk SHA read before `complete-sha-verified.json`. Actual SSH return code0 is required. Failed/partial outputs are retained. There is no Node deletion, signal, GPU lock, training callback or modification. A backup queue receipt proves transfer identity only, not optimizer numerical equality or any of the seven experiments' result.

## Immutable authorization input

Root creates a private JSON byte-identical on Mac and Node; raw human text, QA, weights and tokens do not belong in Git. Shape:

```json
{
  "schema": "opsd-ablation-outer-backup-goal-v1",
  "accepted": true,
  "goal_scope": "seven-group-outer-end10",
  "cluster": "HKUST",
  "study_id": "EXACT_NEW_STUDY",
  "explicit_node_local_with_mac_backup": true,
  "groups": [
    {"variant":"C0", "run_id":"EXACT_32HEX", "config":{"path":"/EXACT/group-C0/receipts/config.json","sha256":"EXACT_SHA"}, "output_root":"/EXACT/group-C0/runs"}
  ]
}
```

The queued outer-completion path/SHA is passed through both remote inventory and stream CLI, saved in the verified closure, and checked before transfer acceptance and before queue COMPLETE (including restart). A changed completion with the same CP cannot substitute its new digest.

The actual `groups` list must contain exactly seven entries in C0,C1,M,R,W,P,J order, all fixed distinct run/config identities. The short example is not a valid default. Publication binds the helper manifest/commit/remote readback/files and either the exact current run or a `run_ids` dictionary covering the seven study runs. Missing config/source/goal bindings are backup HOLD, never a training failure.

## Entries

One checkpoint: `backup.py inventory|stream|backup` retains the reviewed explicit CLI. Bind `--config/--config-sha256`, `--checkpoint/--checkpoint-sha256`, `--expected-run/--expected-updates/--expected-outer`, `--expected-completion/--expected-completion-sha256`, `--goal-identity/--goal-identity-sha256`, helper manifest/publication/commit references. Mac backup also requires `--remote-helper-root/--remote-publication/--remote-goal-identity/--destination`; destination must be new. The remote entry accepts only actual HKUST1 UID/EUID1005 and SSH alias `hkust1`, CVD empty and an independent `backup-...` marker. Inventory and stream write only stdout/stderr.

Autonomous Mac queue after publication:

```sh
python3 -B queue.py queue \
  --goal-identity /PRIVATE/goal.json --goal-identity-sha256 EXACT_SHA \
  --remote-goal-identity /NODE/receipts/backup-goal.json \
  --source-manifest-sha256 EXACT_HELPER_SHA \
  --publication /PRIVATE/publication.json --publication-sha256 EXACT_SHA \
  --remote-publication /NODE/receipts/backup-publication.json \
  --source-commit EXACT_40HEX --remote-helper-root /NODE/source/opsd_ablation_outer_backup_v1 \
  --destination /PRIVATE/new-serial-backups
```

The queue's own Mac lock only prevents duplicate backup workers. It polls read-only discovery every60s (minimum30), requests one full pinned package at a time, and logs failures as BACKLOG with at least300s before a new independent attempt. Previous failed directories and partials are never overwritten or deleted. A restart revalidates the same goal/source identity; completed receipt SHA/actualrc/target and fresh local metadata continuity are checked, without claiming a fresh full tensor audit. All70 successful transport receipts end the queue. `--once` performs one discovery/pass for inspection. Source/config or pin disagreement remains visible in queue error evidence and cannot silently replace a job.

Before every archive the tool measures actual declared core+closure bytes and requires archive upper bound **plus full extraction plus at least5GiB** available on Mac. This is a per-job reserve, not a promise that70 permanent double copies fit. Space/network failures leave a private backlog and do not block C0 or later training. Node unique checkpoints, raw debug files and existing backups remain untouched; no retention permission is inferred.

Focused CPU tests (no SSH/model/scientific forward tests):

```sh
python3 -B -m unittest discover -s experiments/search_e1/control/opsd_ablation_outer_backup_v1 -p 'test_*.py' -v
```

Thirteen fixtures cover changed0–9 scope, legal K0, shared/full-state package, reference closure/dedup, altered closure, unexpected source member, immutable queue pins, independent CPU transport marker, nonzero/partial persistence, measured space requirement, queue-space BACKLOG, changed completion with unchanged CP, and success/restart saved completion pin checks. The USTAR/fixed-handle/extraction/readback functions remain byte-identical to their reviewed predecessor; no broad scientific tests were repeated.
