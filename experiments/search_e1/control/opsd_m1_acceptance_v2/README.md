# M1 final independent CPU acceptance V2

This new standalone source never changes the active 15 diagnostic files,
scientific runtime, process, routing, model, or dataset. It never imports Torch,
models, tokenizers, NumPy, CUDA, or HTTP. Root publishes and verifies the source
before staging/executing it. Run once **after** the actual controller exits.
A running/missing/unknown terminal proof rejects before expensive end hashing.

## Independent V2 ownership correction

The published V1 CPU acceptance rejected an inaccessible `/proc/PID/environ`
for same-UID user systemd / sd-pam and an SSH service session. This was an
infrastructure classification failure, not evidence that the completed M1
model workers were still training. V1 source and its rejection output remain
preserved. V2 is a new directory and new acceptance output; it never hot-edits
V1, the 15 diagnostic files, scientific runtime, or any process.

`owned_process_audit.py` first binds kernel PID/starttick/PPid/comm and the
effective UID from `/proc/PID/status`. The owner of the `/proc/PID` directory
is separately recorded: dumpability can change that owner, and it is not the
kernel EUID. A known owned PID with the same starttick always fails, including
zombies, changed EUID or unreadable identity. Only an absent PID or a different
starttick can supply the separate absence/reuse observation; neither supplies
an exit code.

For every actual same-UID process, an exact owned identity, readable matching
`E1_RUN_ID`, actual original worker/controller script, original run `--output`,
or kernel parent chain reaching an owned identity fails before exclusions.
An original run used only as `--run-dir` / `--trace-root` input is recorded as
an input, not execution ownership. This preserves the independent CPU token
reader. An inaccessible environment remains explicitly unknown. The only
opaque live exclusions are verified predating user systemd / sd-pam (`sd-pam` or actual `(sd-pam)` kernel comm only) / tmux
infrastructure with a root init parent chain, or an actual kernel root-owned
sshd parent service chain. An argv string merely saying sshd is insufficient.
Every exclusion records the kernel identities and reason, without environment
contents or credentials.

Unknown live opaque processes and unknown zombies fail closed. A zombie can
only be excluded by an exact PID/UID/starttick/PPid independent CPU provenance
receipt plus SHA-bound source/launch/rejection evidence, never by an old
starttick alone. Such evidence cannot override owned/run/argv/parent matches.
Production `independent_cpu_exclusion_proofs` is empty: the historical CPU
zombie PID206996 was already reaped and receives no stale production exemption.
V2 writes `process-audit-start.json` and `process-audit-end.json`; the accepted
release embeds both proofs and a fresh SHA/end-stat proof of this scanner.
All original 192-case, 2688-probe, source/weight, actual exit0 and array checks
remain required.

## Fixed production scope and prerequisites

`production-entry.json` binds current run `894021c8f55f436f96bb3cb146ab05bb`,
UID 1005, the existing 192-case source, 15 diagnostic and 305 scientific source
files, 12 source100 model/tokenizer files, and four adapters with two files each.
The output directory must be new. All prior run files remain in place.

Stage the following small, immutable evidence files to the paths named in the
entry configuration, verifying their full SHA; do not copy original QA bodies:

- Live controller/pane binding: local
  `work/search-e1-analysis-output/opsd-debug-plan-review-20261008/m1-controller-live-identity-20261008.json`,
  SHA `f8b1b80ca4301641a2dcb3e006a3a2ba349d2bfe8865ad880fabe181762998e5`.
  Controller 33834 / tick 922457688; pane shell 33832 / tick 922457687; pane `%60`.
- Existing node tmux preservation receipt:
  `/data/shaoshitong/search-e1-opsd-debug-20261008/receipts/m1-tmux-exit-status-preservation-20261008.json`,
  SHA `d872cfc0da2329e319167f6d8d6ae85ac48ce309d21fd83d92c6b75bf61ed45b`.
- Published CPU region helper at the configured **new sibling source** path,
  SHA `55bfd3f74aed1dd5b51bb5fa91244cc3940ecd02ac2c937c3282208942df311e`.
  It is SHA-checked before import, `__file__` checked, bytecode writing disabled,
  and source metadata rechecked after calls. No `run()` entry is invoked.
- Two existing selected-array proofs from local
  `work/search-e1-analysis-output/opsd-debug-audit/`, named in the entry config,
  SHAs `2b5ff309b67ad5371dcc6da3bfe4f6b19731e2a7f3864f9246d014cd2a3914e8`
  and `f7c83e57cd98281d0774c9129c7582937c2588702379c2e6d0ee725c8219fb77`.

The configured region/proof paths do not yet claim deployment. Root stages the
published helper and compact metadata copies. No raw NPY, weights, data, or
installed environment needs transfer.

## Acceptance requirements

1. Startup proof binds config SHA and actual live controller UID/PID/tick to its
   pane shell. The **actual final tmux reading** must retain the same session,
   pane ID, pane PID and start command, with `pane_dead=1` and nonempty numeric
   `pane_dead_status=0`. Controller final must say `WORKERS_COMPLETE`, the exact
   three original children, and each actual child return code zero. A vanished
   process or tmux pane never supplies an exit code. All owned identities must
   be absent or reused; an owned zombie fails. Same-UID run descendants must be
   absent too. This is rechecked after the long CPU audit.
2. Each worker final status is COMPLETE with 64 assigned/complete; actual
   UID/PID/tick/PPID/PGID/GPU/argv identity matches startup. Complete result SHA
   matches producer, indices map by `%3`, results are unique and cover exactly
   192 source cases / six cells of 32. Source semantic hashes are verified.
3. All expected persisted and embedded generation/summary artifacts agree.
   Four fixed versions retain the actual original generation budgets. There
   are **2688 nominal probe slots**, comprising 128 reference-eligible cases
   times 9 and 64 hard cases times 24. Each is complete or an explicit allowed
   empty-response/mask/joint-context skip with an actual boundary reason.
   Full-vocabulary `cap_skip` is a separate, justified status, not a missing
   token probe or an omitted expected trajectory.
4. Completed token files are streamed completely, with original recorded SHA,
   byte/row count, finite JSON values and distribution fields, native mask and
   offset/text alignment, contribution and native loss reconstruction. Hard
   cases retain five failed siblings and null teacher/loss scalars. Original
   response, spans, IDs and masks match across versions; observed teacher q
   invariance checks only identical **preserved** histories. Missing fields or
   legal probe skips are explicit unknown, never full teacher-q verification.
5. Once outside all token loops, fresh CPU reads hash all 15+305 source files
   and 12+8 model files against original manifests/stat identities. Array files
   receive one fresh content hash plus actual NPY dtype/shape/byte-header
   verification. The existing selected-array CPU semantic reconstruction is
   reused only if its per-file SHA matches these final content hashes. Other
   arrays do not gain an independent semantic or full-vocabulary finite-value
   reconstruction claim. No repeated large-array arithmetic is run. End stat
   checks detect input replacement/append/truncation during acceptance.

Outputs are private and contain hashes/counts/numeric proof, never QA/response
text. `progress.jsonl` records CPU stages; `case-artifact-proofs.jsonl` contains
per-case/per-probe proof including explicit skips/unknowns; `release.json` and
`tmux-exit-proof.json` retain the M4 path/SHA reference schema and add explicit
V2 ownership evidence required by `opsd_native512_eval_v2.gates.m1_completed`.
`controller_final`, `tmux_exit_proof` and `preservation_receipt` are full
`{path, sha256}` references, exercised through the actual unmodified M4
V2 `checked_json` and `validate_exit_proof` functions in a CPU fixture. Ordinary
student generation requires `privileged_reference is False`. Recorded protocol
EM is only checked for file/result consistency here; a later frozen QA classifier
independently recomputes all 192 cases. This helper makes no such scoring claim.
`source_and_weight_end_verified` is true only after actual fresh reads. Failure
creates only `rejected.json` plus progress and returns nonzero. A new execution
requires another new output directory; no previous evidence is overwritten.

## CPU tests and node execution

Local fixtures (27: 14 original acceptance plus 13 ownership tests) are small,
independent, and do not prove real M1 completion. From the publisher root:

```sh
python3 -B -m unittest discover -s experiments/search_e1/control/opsd_m1_acceptance_v2 -p 'test_*.py' -v
```

After Root publication and staging, the exact node entry is:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/env/bin/python \
/data/shaoshitong/search-e1-opsd-debug-20261008/source/opsd_m1_acceptance_v2/verify_m1_completion.py \
--config /data/shaoshitong/search-e1-opsd-debug-20261008/source/opsd_m1_acceptance_v2/production-entry.json \
--output /data/shaoshitong/search-e1-opsd-debug-20261008/analysis/m1-end-acceptance-20261008-v2
```

The actual end audit has not been started by the preparing subagent. No
completion or M4-start claim is inferred from fixture success. No new model
logic, scoring change, training update, GPU request, process signal, or restart
is part of this helper.

`SOURCE_MANIFEST.json` binds the seven V2 source/config/fixture/README files.
Root must publish and stage this final independent manifest before execution.
No V2 real-run acceptance is inferred from these local fixture results.
