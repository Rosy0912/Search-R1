# Fresh200 repair review target

The latest human goal is a fresh official Qwen2.5-3B-Instruct actor and reference
at revision aa8e72537993ba99e69dfaafa59ed015b17504d1, with no restored model,
Adam, scheduler, sampler or RNG state from previous training. Historical runs
and their failures remain evidence, and contribute zero updates to this goal.

Each complete outer uses the existing common train split (169028 questions),
512 questions and five actual Tree leaves per question, allocated as three and
two leaves across two roots. No sixth leaf is generated and discarded. Original
inter/intra advantages and equal-leaf objective are retained. The 2560 leaves
with global mini128 and epochs1 require twenty actual optimizer returns per
rank on eight ranks. Actor micro16 (two per rank), old/reference micro32 (four
per rank), LR1e-6, fixed57 complete-outer warmup, entropy0, referenceKL0,
low_var_kl, PPO clip0.2, global gradient clip1 remain explicit requirements.

The independent initial review of 1e06c6 identified three source blockers:

- F01: a stopped outer had diagnostics but no durable partial model/Adam state.
  Matched all-rank safety refusals now return an identical gathered envelope
  after saving real local worker shards, optimizer, scheduler, gradients,
  buffers, module flags and RNG. The driver saves the consumed batch, sampler
  and controller state before raising, without crediting a completed outer.
  The counter advances immediately after optimizer.step returns, before audit
  IO. Nonfinite state is preserved and labelled; incomplete IO is retained as
  INCOMPLETE. Ordinary unmatched failures start no additional collective and
  cannot claim worker writes or saved worker state. There is no rollback or
  supported automatic resume from a partial snapshot.
- F02: sampled-mu provenance omitted best_of and other native controls. The
  collector now reads 24 actual fields, with strict n/best_of/_real_n equal to
  integer1, actual unsupported controls equal to None, and no guessed defaults
  for missing fields. The actual vLLM facade checks before native generation
  as well as after generation; an engine mutation refuses instead of silently
  rebinding controls. Historical records missing controls remain UNKNOWN.
- F03: the quota accessor assumed mapping-style configuration. It now supports
  both mapping and attribute configuration, with affected legacy and exact-five
  CPU regressions. Fresh configuration also requires explicit best_of1.

These repairs are source and synthetic CPU evidence, not actual FSDP/NCCL/Ray
acceptance. The original actor/vLLM probability gate stays unaccepted. Sampled
mu is an explicit, separate objective change; neither a widened tolerance nor
beta/mu importance weighting is silently introduced. Full beta retains the
original reward/reference semantics; monitor beta checks same-partition repeat.

Execution requires a new clean source commit and a context-free strict review
of these bytes, the complete fifteen-file runtime wrapper, and the same-node
preparation/dispatch helpers. Whole official model and split byte identities,
actual Hydra resolution and official-tokenizer historical preflight remain
necessary before zero-write eight-GPU observation. That observation must show
zero optimizer, backward and scheduler writes. Production thresholds require
real observation and short-training evidence; a CPU pass cannot supply them.

The wrapper currently implements zero-write calibration only. Production
training must be implemented and independently reviewed before launch. Its
first five complete outers must pass actual state/HF/reward/resource checks,
then the same living instance continues through outer200. Required saved
steps are5/50/100/150/200; development is512 fixed questions, final is all seven
datasets/51713 questions. Completion also requires actual counters, checkpoint
reload, finite parameters, cost and owned-resource cleanup. Everything remains
on h100-9 with loopback Ray/retrieval and the protected background processes.
