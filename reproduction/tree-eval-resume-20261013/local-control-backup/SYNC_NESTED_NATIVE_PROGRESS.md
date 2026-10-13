# Local native progress evidence publisher

`sync_nested_native_progress.py` reads three explicit existing small local JSONs:

`python3 SCRIPT/sync_nested_native_progress.py --workspace WORKSPACE --poll outputs/tree-nested-branch-handoff-20261013/NATIVE5_CONTINUE_POLL_N.json --native-readback outputs/tree-nested-branch-handoff-20261013/NATIVE_HISTORY_OUTER_METADATA_READBACK.json --resource outputs/tree-nested-branch-handoff-20261013/NATIVE5_CONTINUE_RESOURCE_N.json --outer OUTER`

Paths are literal caller choices, not automatically the most recent glob result.
The tool has no SSH, subprocess, watcher, model, tensor, service or signal action.
Do not run concurrently with another local publisher; only one local status writer
should operate at a time. Preparation/publication of this implementation did not
execute the helper or update any existing STATUS/REPORT/progress receipt.

Identity is fixed to source778b2b43/config355edf52/native5 actual restore receipt
cceb1cfe, target200, same MainPID503255 and Invocationa313bda9 on h100-10. It
checks a full metric for selected completed outer, persisted actual outer credit,
all8 actual rank successful writes, finite/nonzero gradient counts and fixed
sample observations; COMPLETE native writer metadata and all8 finite states,
actual cumulative Adam matching persisted accounting, scheduler N and actual
saved driver next_step N+1. It verifies original/archive manifest SHA agreement
and binds all three local input files by SHA and stable reads.

Twenty opportunities must actually be processed, with at least1 successful write
per rank; success may be1..20. There is no20-writes/20*N/always-zero-skip gate.
Actual metric opportunity skips, explicit per-rank success/skip, partial writes
and no-credit all-skip attempts are kept separate. Missing rank opportunity
fields stay null/unknown; `skipped_empty_minibatches` is never substituted for
all skipped opportunities. The complete full metric supplies actual processed/
skipped aggregate opportunities; rank-specific unknown fields are not invented.
An abandoned or all-skipped attempt cannot satisfy the selected completed outer.
Current16/17 happened to have20 successful writes and0 skip, but that is no
future completion criterion. Profile observations remain separate from dev time.

For N<200 an actual N+1 stage marker and active same unit/PID/Invocation are
required. For N=200 `actual_next_outer_entered` is null; actual final-dataset51713
stage/metric, explicit downloaded final-evaluation report, exact original terminal
or explicit final-only handoff evidence is required instead. `driver.next_step`
is copied and checked as saved metadata; it never claims training201 began.
The normal final phase can remain active or the same Invocation may be honestly
terminal. Goal stays active: final51713/HF/tokenizer/acceptance/cleanup are not
satisfied by this progress receipt.

Periodic dev is current only when an actual complete512 report with finite F1/EM
exists for that same outer. Nondev `actual_dev512` and current dev seconds are
null; latest_dev512 retains only the last actual complete score. Missing dev on
a dev outer is pending/unknown, never borrowed old timing. The all-in step time
is preserved; ordinary training/archive excluding dev is unknown on a dev outer
unless current actual dev time is available. The successful return itself does
not accept profile costs, full-shard gradients or numerical equivalence.

Raw legacy collector pending is preserved under its own key. The actual separate
source-bound watcher state/systemd snapshot is a distinct observation. Missing
native metadata/archive lag emits PENDING and writes no success receipt, STATUS
or REPORT; normal roughly1min settling is not labelled training failure.

Successful publication creates immutable FULL_OUTERN_PROGRESS_RECEIPT.json and
CONTINUE_OUTERN_PROGRESS_REPORT.md, updates both existing STATUS/REPORT aliases
and CURRENT_NATIVE_RECOVERY_FOLLOWUP.json by preserving their ancestry, historical
failed cost, code/publication, future HF/eval deployment and other existing keys.
It mirrors only these small outputs into the two already existing mirror folders.

Older outer counters or earlier poll epochs refuse writing over a newer STATUS
or followup. Existing outer receipt refuses overwrite. Inputs and status bytes
are re-read before publication; concurrent changes return no-write. Atomic local
file replacement is per file, not a cross-file transaction, so an interrupted
write batch must be inspected and reconciled locally rather than blindly retried.
There is no training retry or remote repair path. Missing evidence/rejections
print structured reasons and exit2; JSON/I/O errors exit1. No failure or pending
observation writes a successful receipt, and no exception changes the goal.

At actual200 only, a complete metric200 line is optional when it has not yet
been printed during slow final validation. The same actual200 credit,8 rank
write/finite/nonzero returns, native cumulative/scheduler200/real driver next201
metadata and actual final-phase/terminal/handoff evidence remain mandatory.
Such a receipt explicitly says training updates/native complete but full metric
pending; current metric, timing, variances and unobserved opportunity scalars are
null. It does not wait for slow final scoring, invent metrics or claim training201
entered. If metric200 is available the full existing validation rules apply.
Every N<200 still requires its complete metric and real next-outer stage marker.

REPORT prints the latest actual dev outer/F1/EM even on a nondev outer, while
current actual_dev and dev seconds remain null. Current metric variance and
candidate/token/retrieval cost scalars are printed only from that outer's metric;
unobserved values remain null. No previous dev time or variance is recycled.
The immutable actual outer18 receipt remains tied to its original f720... helper
SHA; this later implementation applies to subsequent explicit invocations only.
