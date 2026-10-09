# PR100 aggregate static plot

CPU-only plotting for a completed snapshot from `opsd_pr100_reports_v1/report.py`.
Requires Python and matplotlib. Reads only its five small aggregate files; writes
one PNG and one provenance JSON to a **new** output directory. No SSH, network,
model/raw/token/tensor reads, scoring, bootstrap, training, or GPU work.

```sh
python3 plot.py --report-dir /absolute/path/to/accepted-report --output /absolute/path/to/new-plot
```

The four panels show sampled pre-update TRAIN EM on different cohorts, recorded
equal-trajectory raw/post-clipping loss per completed outer, loss for each real
Adam update, and only closed fixed512 validation EM at 10/20/...100. TRAIN EM is
not per-Adam EM or a heldout curve. No smoothing, interpolated points or partial
512 evaluation rows are used. Source100's horizontal baseline is shown only
when the accepted report records its actual zero-summary path/SHA; its value is
taken from that report, never filled from a remembered result.
The validation legend sits in the upper right to keep point labels readable.

The helper checks the small input artifact hashes against `complete.json`, and
records exact input/source/output hashes and completed coverage in provenance.
It renders already accepted counts rather than re-verifying model or runtime
science. The original first10 plot and report stay untouched.
