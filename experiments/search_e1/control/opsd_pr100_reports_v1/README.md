# PR100 recorded curves

Standalone stdlib CPU reporter for the new PR100 run. It reads only the explicitly SHA-bound training and periodic pipeline configs, accepted completed outer prefixes, their retained checkpoint JSON manifests and actual committed update receipts,the original accepted2560 strict0/1 reward aggregate in immutable sampling stats per completed outer, and closed native512 curve/boundary/summary/actual-exit records. It does not read raw rollout JSONL, tensor payloads, optimizer state, token loss JSONL/vocabulary distributions, model weights, tokenizer or private per-question evaluation records. It does not recompute a scorer, bootstrap, generation, gradient or training state.

```text
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 <original env python> -B <helper>/report.py --training-config <absolute config> --training-config-sha256 <SHA> --pipeline-config <absolute pipeline config> --pipeline-config-sha256 <SHA> --output <new unique remote analysis directory>
```

The output must be inside this experiment's HKUST `/data` tree. Every invocation writes an independent snapshot, without replacing original outputs or an older report. No GPU locks, signals, HTTP, subprocesses or SSH are used. Root publishes this helper before executing it on the node; code/config and small aggregate outputs may be kept locally, but raw inputs remain remote.

Outputs:

- `train-outer-curves.csv`: each completed outer's pre-update sampled TRAIN EM from the original accepted all512×5 strict reward counts, eligibleP/usableK, actual/cumulative Adam, valid tokens and equal-trajectory recorded raw/clip loss. Each outer has a different TRAIN cohort; this is neither a fixed test curve nor pass@5.
- `committed-adam-loss.csv`: each actual committed Adam update's recorded equal-trajectory raw/clipped objective, true tail denominator, token count, norm and committed retry count. No per-Adam EM is invented.
- `native512-curves.csv`: each closed10-outer boundary's fixed512 hits/EM, NQ256/Hotpot256, gained/lost/preserved/still-wrong against the existing source100 baseline and its accepted native10000/seed42 paired bootstrap interval. No extra baseline or bootstrap is run.
- `summary.json`, `report.md`, `complete.json`: public aggregate coverage, limitations, source summaries and artifact hashes. Partial prefixes remain explicitly partial; no unfinished outer or partial512 predictions are included.

These reports summarize existing scalar evidence. They do not constitute another tensor audit, reproduce original per-token loss, claim exact degradation between ten-outer checkpoints, or establish algorithm causality. Compact inputs/output only; no new tests or reviewer round is part of this helper.
