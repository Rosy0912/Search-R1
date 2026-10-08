# Independent root integration review

Review scope: root-owned profile/config, startup, holdout integration, controller,
trainer, Fresh200 ledger and scheduler integration, and the other agent's actor
and generation integration. This reviewer authored `bapo.py`,
`baseline_alignment_data.py`, and their protocol/data tests; those components
are **not represented as independently reviewed** here.

## Findings at the initial integration snapshot

1. **P1 — baseline loss reduction differed.** The initial aligned config and
   `validate_aligned_config` mandated `sequence_mean`. Search-E1 R1 uses token
   means within each logical microbatch (`archive/upstream/Search-R1/.../dp_actor.py`),
   and `control/physical_micro.py` preserves logical token weights and the
   virtual four-rank pairing. Equal weighting of trajectories changes PG and
   actor KL for unequal response lengths, beyond the prescribed BAPO changes.
   Root/actor agent subsequently changed the aligned profile and collective
   binding to `token_mean`. Actor PG, entropy and KL now divide by each fixed
   microbatch's masked-token count and then by the accumulation count. The added
   unequal-length actor comparison tests this reduction against archive code.

2. **P1 — dev512 could silently become dev32.** A composed aligned config with
   `data.val_data_num=32` passed `validate_fresh_config`, returning `(200, 57)`.
   The dataset constructor then samples validation rows to 32, despite exact
   manifest verification of the original 512-row file. Similarly native-only
   diagnostic mode passed aligned validation although its required native
   sampler tensors are deliberately absent. Root has now added null train/dev
   subsampling guards, a diagnostic-mode prohibition, and exact dev512,
   final51713, stage64 count gates.

3. **P1 — actual sampler KL diagnostic is unreachable.** Aligned rollout emits
   `raw_sampling_diagnostics` as non-tensor provenance, and deliberately does
   not attach raw selected probabilities to retokenized action tensors. The
   actor branch producing `sampling_policy_diagnostic` requires tensor keys
   `sampling_log_probs` and `sampling_mask`; actual aligned packed trajectories
   never contain them. Consequently real aligned runs have HF-old KL/tail
   diagnostics and raw sampler captures, but not a numerical KL comparison to
   the actual sampling policy. Renaming HF KL is not evidence of sampler KL.
   **Resolved honestly:** aligned monitor now removes the misleading sampler-KL
   name, reports `sampling_policy_kl=null` with status
   `unavailable_on_retokenized_training_actions`, and separately identifies
   frozen HF KL and raw sampler captures. No raw-token-to-retokenized probability
   substitution is claimed. The subsequent bounded raw-context diagnostic now
   computes native-vs-current-HF K3 gaps and negative-advantage token-tail
   fractions on original native request/completion contexts, using verified
   effective sampler proofs. These numbers include cross-backend drift and are
   explicitly not labeled pure policy KL or used as update brakes.

4. **P1 — retrieval identity remains unproved.** Existing
   `preflight.validate_assets_and_retriever` checks response structure and
   candidate count only. A service with another encoder, corpus, or index passes.
   Aligned topk3 and identical formatting/truncation code are not evidence of
   matching local GPU E5/corpus/index. A new identity gate now binds independently
   pinned encoder/corpus/index/service bytes, live loaded/current file identities,
   encoder parameter device/dtype, inspected GPU FAISS children/options, and
   ordered baseline golden document/format/token probes. Six retrieval fixture
   tests pass, including live-object CPU-vs-config and actual child-device
   inspection. No module defect found. Root startup/config wiring subsequently
   landed: aligned asset preflight loads the actor tokenizer, verifies the bound
   retrieval identity/probes, and prints its receipt before Ray worker allocation.

5. **P2 — legacy E1 gate regression.** The first trainer patch made
   `_e1_pause_after_five` depend on `fresh200_gate_enabled`; historical E1 configs
   need their original separate gate. Root has restored the original E1 condition.

## Verified integration behavior

- Fresh200 aligned ledger expects 20 committed Adam writes per rank at outers
  1, 49, 50, and 200. Its cumulative no-skip budget is 4000 per rank. Aligned
  acceptance requires `COMPLETE`, zero skips, and collective rank agreement.
- Worker scheduler advances once after a collectively complete outer. Fatal
  update state does not advance it. This differs intentionally from per-mini
  scheduler stepping; no switch to ten writes remains in aligned ledger paths.
- Trainer recomputes HF old probabilities before actor updates and binds a
  detached snapshot. Aligned trainer bypasses `bind_actual_behavior` and the
  native-sampler first-batch gate. Reward-level KL remains zero; reference KL
  is applied in actor loss with coefficient .001.
- Default aligned gate is disabled at outer5; original profile defaults remain
  inherited except for the explicit aligned config. Save-at-five still occurs.
- Training and aligned validation share the generation budget and aligned
  engine. Validation uses greedy generation; final evaluation switches to the
  pinned final files after exactly 200 completed outers.
- Shared dataset/order manifest integration verifies before worker startup and
  rechecks unchanged datasets at exit. Actual emitted question IDs are audited
  before sibling repetition. The order manifest must come from actual baseline
  indices or its captured pre-iterator RNG state; a nominal seed42 claim alone
  cannot establish historical baseline order.

## Verification evidence and limits

Workspace interpreter: `work/fresh200-local-test-env/bin/python`.

Independent integration regression command covered:
`test_baseline_aligned_actor.py`, `test_baseline_aligned_generation.py`,
`test_baseline_aligned_validation.py`, `test_baseline_raw_sampling_capture.py`,
`test_fresh200_driver.py`, `test_fresh200_contract.py`, and
`test_fresh200_driver_config_transport.py`: **36 passed** at the initial actor
snapshot. Production configuration composition was also exercised for the
subsampling/diagnostic counterexamples. These tests do not establish live
eight-rank CUDA/FSDP behavior or a real local retriever identity.

Final re-review additionally covered `test_baseline_alignment_contract.py` and
`test_baseline_retrieval_identity.py`, with the updated actor test using a
float64 analytical fixture matching the kernel's intentional float32 surrogate
conversion: **46 passed**. This includes all-200 ledger accounting at 4000
committed writes per rank, no legacy mini-size transition, HF snapshot binding,
production config guards, 13 actor tests, raw capture/greedy validation fixtures,
and six fail-closed retrieval identity tests. Repository `PYTHONPATH` was set
explicitly to the new isolated checkout. No GPU parity is claimed by these tests.

## Raw diagnostic and launch-entry follow-up

The bounded raw diagnostic reconstructs original native request/completion
contexts, uses at most four retained rows and five turns per rank, pads to equal
FSDP forward counts, and reports a cross-backend context gap rather than pure
policy KL. Nonfinite raw diagnostic probabilities return unavailable without
skipping any of the twenty training writes. Its synthetic actor-wiring fixtures
initially failed because a directly imported `distributed` alias was not covered
by the CPU fixture's monkeypatch; these fixture failures do not establish a
production defect.

Two integration findings were sent to root and resolved before final acceptance:

- The raw module requires native `sampling.logprob_semantics`, but the actual
  raw capture metadata did not supply that field. Synthetic diagnostic fixtures
  supplied it manually. The actual callback now binds native output provenance
  and the observer inspects the actual enqueued sequence group after the original
  engine `add_request`. Request ID, request tokens, effective sampler values and
  evidence hash are bound to the captured native trace. The positive test executes
  the real rollout capture callback, prepares its journals with the real helper,
  and computes a finite raw-context diagnostic using model selected-token
  probabilities. Effective temperature/support transforms, missing group APIs,
  and inspection failures produce unavailable diagnostics while generation
  tensors, requests and sampler behavior match ordinary generation.
- The launch entry protects `checkpoints` and root `train.log` only; prior
  `logs/train.log` or early-failure diagnostics can remain without either marker.
  The entry now calls `aligned_startup.claim_new_run` before preflight. It requires
  a canonical independent `runs/<new-id>` path, rejects symlinks and unrecognized
  prior artifacts recursively, permits only named resource-watch bootstrap logs,
  and creates its startup claim using `O_EXCL`. Duplicate claims and nested prior
  training logs are tested. The entry correctly refuses dirty source and requires
  actual inherited UUID leases and live own-supervisor ancestry. Resource-watch
  deployment remains an explicitly documented external prerequisite.

Final focused re-review after those fixes: **52 passed** across actual raw capture,
raw diagnostic helper/real two-rank Gloo, actor, config/ledger/startup, and retriever
identity suites. The positive raw capture-to-helper test does not fabricate the
native trace's proof fields; they come from the production callback and observer.
Real Gloo fixtures check equal bounded forward-slot counts and collective
unavailability. Other actor fixtures verify nonfinite raw diagnostics still leave
all twenty training Adam writes intact.

## Final code-only verdict

**ACCEPTED for the reviewed source scope.** No remaining must-fix code finding in
root integration, actor/generation integration, bounded raw diagnostics, retrieval
identity wiring, or exclusive new-run entry. Initial findings above are closed.
This acceptance does not independently review this reviewer's own protocol/data
implementation, and it does not approve a training launch, establish target GPU
parity/performance, verify a live baseline retriever, or claim experimental effect.

Signed: `/root/alignment_protocol_data_review`, 2026-10-09, after the final
52-test focused run with the new checkout on `PYTHONPATH`.

The reference KL kernel bounds its input before exponentiation but retains the
baseline low-variance kernel's final upper clamp10. PPO log-ratio clamping and
Dual-Clip are disabled in the aligned surrogate. Raw sampling probabilities
cannot legally be substituted for retokenized HF denominator probabilities.

No baseline checkout, running job, remote service, or source publication was
changed by this review. Root owns final integration fixes and publication.
