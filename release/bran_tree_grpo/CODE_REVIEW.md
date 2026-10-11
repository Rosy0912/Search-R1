# Independent release review

STATIC_SOURCE_RUNTIME_REVIEW_ACCEPTED

Reviewer: context-free subagent bran_release_review (fork_turns=none),2026-10-11. Read-only local static review; no remote access, submission or training.

No newly introduced release-blocking code issues found.

- search_r1 and verl match the original branpo-source checkout byte-for-byte; core diff against frozen8b6f99d1b3353145200eb3fa7c16a579a79a987c is empty.
- All480 deletion paths match CLEANUP.json. Surviving production code has no executable dependency on the deleted files. retriever_local.py and the backed-up owned runtime helper match the external production pins.
- pinned_files.read_regular_bound is identical to the original function; guard_entry.py and submit_once.py imports are updated. The package builder includes the renamed module.
- The launch chain is build_package → make_release → render_dispatch → submit_once → launch.sbatch → guarded_preflight_main → supervisor. UUID leases, own cgroup, resource/storage guards, fixed split and official model identity checks remain. Submission journal/lock and unknown-result refusal remain.
- Shell syntax and runtime Python AST checks pass.

This file must be committed/pushed with the release before executing the launcher. The launcher depends on retained18's existing pinned helper/environment/assets. Those remote prerequisites were not checked in this review; this is not an empirical training acceptance claim.
