# Read-only G1 reward/probe analysis, 0–150 iterations

`findings.md` separates actual TRAIN coarse raw-reward evidence from matching-head calibration reconstruction. Raw features, weights, complete rollouts, and node logs stay remote.

The `read_*.py` / archive extraction helpers are executed on the recorded node with its original environment; they emit only compact CPU summaries and source identities. They do not mutate training or generate new model samples. The corrected group reconstruction uses model-authored search counts; the superseded actual-call-cost output is retained privately as unaccepted evidence.

Run `python3 build_analysis.py`, `python3 build_raw_reward_analysis.py`, then `python3 build_canvas.py` here to rebuild the graphs. Python needs NumPy/Matplotlib locally for plotting only. Remote probe reconstruction also needs the original sklearn/joblib environment. `probe-reward-evolution.canvas.tsx` contains aggregate values only.

G1 actual raw reward vectors were not saved. Neither global averages nor the selected calibration cohort can replace all historical 2560 individual TRAIN rewards per iteration.
