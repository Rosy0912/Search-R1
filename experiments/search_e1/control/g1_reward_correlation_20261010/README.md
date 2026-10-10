# Search-G1 designed reward versus strict EM correlation

**Correction: the user's requested full individual TRAIN trajectory Pearson has
NOT been computed.** Prior iteration-mean and selected calibration coefficients
are withdrawn as answers to that request. They are retained solely to audit the
statistical-unit error. Do not report them as full per-iteration trajectory or
full 0–150 pooled correlation.

The required input is one paired original/shaped reward per actual TRAIN
trajectory, joined by run/iteration/trajectory ID, without collapsing groups or
iterations before computing Pearson. Historical reconstructability is under
audit; absence of direct reward fields does not by itself imply that saved tokens,
masks, data and historical frozen evaluators cannot recover the pairs.

Raw is explicitly defined here as original strict correctness C=0/1. Designed R is G1 shaped reward before GRPO normalization; they are not two names for the same tensor.

The CPU-only reader reconstructs matched-head calibration trajectories at rounds0/50/100 using the embedded SHA-pinned earlier reader. It emits aggregate correlations only. Features, model weights and individual rollouts stay on the original node. Same61 question comparisons are kept separate from differing complete-cohort coverage. Whole-iteration mean correlations use prior accepted compact TRAIN records and rounded console EM, not historical individual trajectory correlations.

All-correct/all-wrong groups have constant EM, so group Pearson is undefined. High R/EM correlation follows partly from the correctness gate; it does not validate b/d fidelity or rule out proxy exploitation. Positive correlation is not proportionality.
