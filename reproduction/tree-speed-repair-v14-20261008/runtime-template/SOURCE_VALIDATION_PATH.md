# Validation source API audit (4ff base; current frozen f7e closure)

This is a source-only finding; no dev512/final51713 or tensor/model/GPU execution occurred.

`search_r1/baselines/controller.py:242-246` passes behavior_mu only to training
SearchRollout. Its `token_preserving_validation_manager` at426-434 does not
pass behavior_mu; `rollout.py:70` defaults it toNone. Only its295-312 branches
compose/validate the training μ tensor. `ray_trainer.py:633-635` explicitly
uses recompute_log_probFalse/do_sampleFalse/validateTrue during search eval.
`vllm_rollout.py:162-173` retains original greedy temperature0/best_of1/n1.
Raw capture's24 attributes in `tree_raw_tokens.py:53-70` permit temperature0.
Temperature1 support belongs to `tree_behavior_mu.py:25,59-64,101-106`.
Thus global enabled training μ does not route greedy evaluation through a
training probability denominator or require stochastic μ for greedy samples.

The inherited cost wrapper had counted validation rows only in
LLMGenerationManager.run_llm_loop. The real token-preserving Adapter calls
SearchRollout.run directly; that class has no Manager.run_llm_loop call.
This draft observes the actual Adapter.run_llm_loop instead, counts its
submitted question rows exactly once, retains all original calls/returns, and
restores the factory in finally. Other manager paths retain their old counter.
Observer.install also uses production's absent calibration flags and new
own-run prefix. These changes only affect external evidence publication.

Validation still shares fail-closed raw-token assumptions: prefix decoding
must not contain UTF8 replacement characters or be nonmonotonic (raw helper
119-122); retained full context and policy/semantic response caps must agree
(trajectory_projection157-174). Real Unicode/rolling/cap refusals cannot be
quietly filtered or marked accepted. This audit therefore proves that μ's
temp1 gate is training-only, not that all512/51713 examples have run successfully.

Old sampler/actor consistency remains unaccepted. No probability/kernel,
evaluation sampling, Tree weight, reward or model-budget change is claimed.

The real SearchRollout._advance at155–161 pads and directly calls
wg.generate_sequences, bypassing Manager._generate_with_gpu_padding. The
observer therefore temporarily wraps the actual _advance to record active
state count in a stack, and reads real submitted worker rows inside the
existing WG wrapper. It records the difference as actual padding and restores
all hooks in finally. CPU fixtures execute the actual Adapter and _advance
function AST up to its native WG call; they intentionally stub subsequent
response/parser/reward processing. Initial512/step5dev512/final51713 input
counts and final51720 submitted rows/7 padding have been tested using tiny
synthetic arrays, including a native failure/restoration case. This is not
real validation, raw emission, GPU or reward-result proof.
