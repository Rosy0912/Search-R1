# Seven-group recorded curves

Reuse the already published `opsd_ablation_outer_report_v1/collector.py` on the exact seven-group goal. This local formatter consumes its accepted, SHA-pinned output and writes 70 explicit slots, a per-group summary and standalone PNG/SVG plots. Pending slots stay blank; partial groups use dashed lines. It does not write to HKUST, sample, rescore, read optimizer tensors, or alter running code.

Unlike the original C0-only formatter, this keeps each group's precision, mixture and actual weights. It does not impose C0's teacher=mixture or weighted=unweighted equalities on M/J/W. Cross-outer means use actual pair count K, including tail updates; they divide neither by Adam updates nor weight sum. Missing metric coverage remains NA. Counts come from the producer's accepted completion/checkpoint/update chain, not a new tensor audit. Raw KL, clipped contribution objective and weighted objective retain distinct labels. Eligible pair classes have denominator P; actual training exposure has denominator K.

The plots summarize different TRAIN cohorts and update-before-sampling policies. They are not fixed held-out512 performance curves, causal evidence, confidence intervals, or complete historical overflow statistics. Final report needs all70 rows, all11 native512 model results and separate debug/persistence/execution proofs. No extra scientific fixture or GPU test is introduced by this artifact formatter.

Publish and read back the source before running it. CLI: `derive.py --facts <accepted facts> --facts-sha256 <exact SHA> --output <new directory>`. An existing output directory is refused. Plotting uses the existing local Matplotlib installation; nothing is downloaded.
