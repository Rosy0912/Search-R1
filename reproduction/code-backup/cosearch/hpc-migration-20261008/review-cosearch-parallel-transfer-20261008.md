# Independent four-stream migration review — 2026-10-08

Decision: no newly identified four-thread startup blocker for this fixed manifest, provided the planned handoff actually proves both the old task process group and all target receivers have exited before starting the new owner. This review did not initiate a transfer, contact another session, or alter production scripts.

Reviewed current worker SHA256: `c7b617ce714ed31db812e0b74f9adf133576b09e83f80ee79a4c98092489ea81` against `migrate_cosearch_hpc_serial_20261008.py`, shared receiver, integration gate and design. The actual manifest has 47,340 file/symlink entries: all paths are canonical and unique, with no path colliding with another path's `.migration-partial`. Four tasks therefore cannot target the same partial file. Large files are partitioned from the serial small bundle; the final full-manifest SHA remains mandatory.

The receiver retains explicit partial `os.write` handling, bounded EIO retry and fsync. Retry obtains the actual remote partial size rather than trusting status. A file enters `verified` only after a pre-existing complete file's full SHA passes, or after exact size/full SHA and rename; partial/progress bytes are not verified bytes. Remote assertion failures now hard-stop rather than retry a corrupted complete/partial file forever. Thread-local entries and RLock protect shared active/verified/retry state and atomic progress writes.

Remaining operational risks, inherited substantially from the serial transport:

- Blocking pipe reads/writes and process waits have no transport deadline. SIGTERM only sets an event checked after a read; it does not guarantee prompt receiver termination. The documented old-receiver quiescence check is essential; a local process exit is insufficient proof.
- Automatic per-file retry has no remote per-partial lock. If an abnormally disconnected SSH session leaves a receiver alive, the new append receiver could race with it. Prefer a remote receiver flock, plus validating the partial offset under that same lock, or explicitly verify receiver quiescence before retry. Final SHA prevents false completion but does not prevent wasted/corrupted partial work.
- `active_files.partial_bytes` measures bytes supplied to the local pipe, not remote durable bytes; report it as advisory. No acceptance logic currently mistakes it for verified completion.

The local integration gate exercises four streams, disconnect/resume and corrupt complete-file refusal using local shell routes and an injected test receiver. It does not prove SSH remote process cleanup or shared-storage EIO behavior. Keep remote cleanup proof separate from this gate.

## Receiver-lock follow-up

The final receiver adds a nonblocking per-partial writer flock before opening append, then validates actual partial size against the supplied source offset under that lock. The lock remains held through all writes and fsync. This closes concurrent append between new receivers and stale-offset restart races; old serial receivers still require explicit handoff quiescence because they may not participate in this lock protocol. The integration gate was independently rerun including real receiver stale-offset refusal, held-lock refusal and exact append.

- `migrate_cosearch_hpc.py`: `c7b617ce714ed31db812e0b74f9adf133576b09e83f80ee79a4c98092489ea81`
- `cosearch_transfer_io.py`: `fbbbc76cd0c9061dfd4ee50317b5f629cd65a94d627dabfab2b7903ee914d902`
- `test_cosearch_parallel_transfer.py`: `cbfd4269e0b30f482a346734b23c3605d5c4cce04407bea7da6073dbd6e7db07`
