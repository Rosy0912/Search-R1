# Independent CPU-merge boundary recovery

The accepted online run retains all original scientific code, config and
checkpoints. This independent entry loads the published wait runner from commit
`2cbcc1b4754eff51a52b213240579f5db0e10f87` (manifest
`baa072189d0213309d72dd9d04c67bdaf03361e7e7b452b137040d4bfee9a67f`),
then loads original controller/helper7cf exactly as before. It inherits the
wait-recovery subclass. Only the **merge call boundary in this new process's
loaded controller module** is temporarily replaced, restored in `finally`.
Original `sampling.merge_latest`, `Controller.run`, mining, rollout, optimizer,
loss, checkpoint and frozen305 runtime bytes remain untouched.

Why: the existing controller660173 has a real500MiB CUDA context on GPU0 after
its in-process PEFT merge. Its exact original release gate therefore correctly
holds. We do not exempt controller-owned GPU memory or loosen any release floor.
Independent proof:
`work/search-e1-analysis-output/opsd-online-audit/controller-cuda-residue-independent-1791452531.json`,
SHA `327b34f6a5b0d43637ba331ef5e3e6a50a4f79a35824c7d296835b2b14152fc9`.
That observation shows the context; PEFT device discovery is the suspected
initialization site, not a traced CUDA call proven by the snapshot.

The fresh controller has `CUDA_VISIBLE_DEVICES=''` and never imports Torch or
model libraries. Original sampling/training spawn explicitly overrides rank
GPU/eight GPUs, unchanged. Each merge launches a registered **CPU** child with
empty CUDA visibility before imports and the same original numerical environment.
The child calls the original merge function exactly once with identical arguments
and BF16 PEFT method. Inspection wrappers preserve original PEFT calls/objects
and verify actual PEFT/merged model parameter devices are all CPU. CUDA is
uninitialized and visible device count0 before/after. Original source12 hashes,
adapter/checkpoint identity and loaded metadata are verified by the original
function/checkpoint helper. The child checks new source publication before model
imports and explicitly rechecks it after the original merge; the inherited
sampling/training terminal wait also revalidates source bindings later.
The parent reads back actual child exit/receipt, all output file hashes/inventory,
unchanged original model identity hash, latest adapter tensor SHA and CP updates.
It returns the **exact original merge identity**, with no added scientific fields.

CPU-child terminal evidence/logs are exclusive private files under
`BASE/receipts/cpu-merge-recovery/UUID/`. Actual `wait()` returncode is durable
before rejection. A nonzero or rc0 without receipt is rejected after strict fresh
GPU/full-owned release. Persistent ownership/admission/evidence I/O uncertainty
keeps all original leases held. The CPU merge boundary sends no signal/kill,
starts no tmux, and adds no GPU phase. The full parent entry still inherits the
original run and launches its original eight-GPU sampling/training phases.
Existing sampling/training use the unchanged2cb wait recovery.

Recovery requires:

* SAME run `1d1d67b60f0e46a2afc0536a36ab69b1`, explicit `--resume`;
  immutable config SHA `d0c30df9b035e73d87bb5402bbcd32c16b2314a4180c451a543522ca5b321b17`.
* Original terminal612 receipt remains bound and freshly rechecked (unknown old
  wait rc remain unknown).
* New previous-owner proof schema `opsd-online-cpu-merge-parent-terminal-v1`,
  accepted/run/config, `old_owner={pid:660173,uid:1005,starttick:924349236}`,
  `old_source_commit=2cbcc1b4754eff51a52b213240579f5db0e10f87`,
  `old_owned_processes` complete unique `{pid,uid,starttick}` list, all freshABSENT,
  `resume_checkpoint={path,sha256}` and `merge_preservation={path,sha256}`.
  The latter checked receipt must be accepted and bind the same run/config.
  Root records actual previous tmux status (143 if that is its true status);
  this is separate from original controller612 failure1.
* Exact committed CP11 is freshly checked through original `check_checkpoint`,
  including eight state file hashes, original source/runtime/adapter and completed
  outer0 cursor. Actual CP11 manifest path is
  `BASE/runs/checkpoints/update-00000011-outer-00000/manifest.json`, SHA
  `464585451581b432d83a97ad96b0b8cc49952c89c3cc39223b6fb5b8fae4a6ad`.
* Root preserves the old outer1 `merged-model.json` and `merged-model-cache`
  outside their original paths. Both original paths must be absent at startup;
  the untouched original run then invokes the new CPU boundary and recomputes
  from CP11. Questions/selection and old empty attempt remain preserved. There
  is no restart from zero and no old Adam reset.
* Original gate/base+four shared leases stay in force. Private18004 oracle and K4
  are not changed. Root alone handles prior660173 narrow retirement and deployment.

The startup outer1 guard is deliberately **one-time recovery-specific**. If this
new runner later fails after committing its new canonical CPU cache/receipt,
another restart needs a new previous-owner terminal proof and independent review
of the accepted CPU receipt/checkpoint, with a new immutable recovery copy as
needed. Do not delete that new canonical cache to reuse the historical660173
absence proof. Do not infer a new controller's terminal status from the old
receipt. This entry does not silently generalize its source-specific CP11 guard
into an automatic recovery policy for later faults.

The new external publication follows the prior receipt format: accepted,
branch`search-e1`, source_commit40hex=remote_readback_sha,
source_manifest_sha256, SAME run/config, and `files` list of actual Git/node
readback path/SHA/bytes/equaltrue for this directory's **all** files including
source-manifest. Prefix is
`experiments/search_e1/control/opsd_online_cpu_merge_recovery/`.
Prior wait publication/manifest are independently checked too.

Parent entry (root fills actual new publication/path/SHA; original environment
retained including `CUDA_DEVICE_ORDER=PCI_BUS_ID`):

```sh
E1_RUN_ID=1d1d67b60f0e46a2afc0536a36ab69b1 CUDA_VISIBLE_DEVICES='' \
  ORIGINAL_PYTHON -B NEW_CPU_DIR/recovery.py parent \
  --config BASE/receipts/config.json \
  --config-sha256 d0c30df9b035e73d87bb5402bbcd32c16b2314a4180c451a543522ca5b321b17 \
  --run-id 1d1d67b60f0e46a2afc0536a36ab69b1 --resume \
  --wait-root BASE/source/wait-recovery-2cbcc1b/opsd_online_wait_recovery \
  --wait-publication BASE/receipts/wait-recovery-publication.json \
  --wait-publication-sha256 d57387b3b495e84589eed9e48c49462dda3df8548701d577b7c30e1f79ae25d1 \
  --source-manifest NEW_CPU_DIR/source-manifest.json --source-manifest-sha256 NEW_MANIFEST_SHA \
  --publication BASE/receipts/cpu-merge-recovery-publication.json --publication-sha256 NEW_PUB_SHA \
  --original-terminal BASE/receipts/wait-recovery-original-terminal.json \
  --original-terminal-sha256 b8978990c3c276570383e82f280eef0d52a7f39d3b73992f8e11e595e8a0484a \
  --previous-terminal BASE/receipts/cpu-merge-parent-terminal.json --previous-terminal-sha256 ACTUAL_TERMINAL_SHA
```

CPU fixtures run without model/data/GPU/HTTP/remote activity:

```sh
python3 -B -m unittest discover -s experiments/search_e1/control/opsd_online_cpu_merge_recovery -p 'test_*.py' -v
```

For isolated node deployment set `OPSD_ONLINE_TEST_WAIT_ROOT` to the frozen2cb
directory and `OPSD_ONLINE_TEST_HELPER_ROOT` to original7cf helper. Twelve tests
exercise the **actual unchanged original merge function** with tiny CPU mock
models, exact call args/BF16/device_map/save method, restored PEFT descriptor,
CPU/CUDA device guards, actual childrc/nonzero/missingreceipt, original modelhash
tamper refusal, persistent owned-release HOLD, parent-no-Torch, immutable config
and fresh prior-owner/proof binding. These are CPU boundary checks, not a claim
that real model merge or next online outer has passed live acceptance.
