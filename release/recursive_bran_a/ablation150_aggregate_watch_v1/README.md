# CPU-only native aggregate watcher

This is a separate reporting helper. It never imports the training entry or
Torch, reads tensor/checkpoint contents, queries GPUs, contacts a service, sends
signals, takes GPU locks, or changes production source/configuration/receipts.
It reads existing small JSON, complete native scalar lines, and exact `/proc`
owner identities. A report is built by the existing `ablation150_reports_v1`
collector from a temporary view containing only closed, contiguous native
outer records and completed eight-rank Adam receipts; the view is removed.
Evidence is cached in the watcher process by immutable config/current entry,
completed-outer file metadata, terminal/wait receipts, study acceptance, and
known small manifest/Adam-file metadata. Growing live logs alone do not trigger
rescans of previously accepted work. Exact owner liveness is refreshed cheaply.
If closure is flushed just before its complete native step line, an active
owner's pending milestone is retried until both records are present.

From a separate published source checkout, on the machine with the evidence:

```sh
python3 -B -u -m release.recursive_bran_a.ablation150_aggregate_watch_v1.watch \
  --base /dataset/shaoshitong/recursive-ablation150-20261014 \
  --output /dataset/shaoshitong/recursive-ablation150-20261014/analysis/aggregate-watch-v1
```

`--once` takes one bounded observation. The normal watcher repeats every 60
seconds. It creates `milestones/<label>/outer-005`, `050`, `100`, and `150` only
when that milestone's actual native dev512 and hashed post-evaluation manifest
are closed, with all preceding native/Adam records present. Existing reports
are hash checked and retained unchanged. Every poll replaces only its own small
`watch-state.json`, which records the required 44 reports and accepted groups.

The eleven labels consist of seven new trainings and four matched reuse labels.
`generation-1` reuses C1; budget10, uniform position, and generation4 reuse the
retained A aggregates. The supplied 8 KB A pack records actual hits
96/199/236/246 at 5/50/100/150, original native log paths, exact captured-line
hash conventions/hashes, and hashes of the retained capture and summary. It
does not invent a common A config identity across initial/resumed native entries.
Reference cost scalars describe only recorded boundary outers, not prefix cost.

Milestone `goal_complete` is always false. Overall completion requires all 44
reports, all seven complete150 receipts, exact current entry exit0 and parent
wait0, accepted context-release receipts, all four matched study reuse receipts,
the full terminal study status, and the current study wrapper's actual exit0.
A retained parent wait1 from an earlier PID/time is explicitly ignored; only a
current bound failure or malformed evidence causes `ATTENTION_REQUIRED` (exit2).
A terminal-looking study status or complete report subset cannot complete the goal.

Scheduler/finite/Adam checks refer to recorded native receipt scalars only;
the helper does not claim an independent tensor reload. Recovery of C1 outer1
excluded original pre-fault sampling from its replay timers, so all affected
milestones retain an explicit missing end-to-end cost coverage marker. EM remains
the existing complete native cohort aggregate: no per-question capture,
paired analysis, bootstrap, A replay, or new evaluation is requested.

Focused fixtures run without training dependencies:

```sh
python3 -B -m unittest release.recursive_bran_a.ablation150_aggregate_watch_v1.test_watch
```
