# CoSearch bounded parallel migration

Human authorization: 2026-10-08 user explicitly requested multithreaded parallel transfer of the already authorized CoSearch assets to HPC3. Existing same manifests, target path, source retention, shared output policy, 1 TB account budget, strict existing routes, no credential or GPU changes remain.

Use four file threads in the primary worker. A single process flock owns the primary manifest; validated unique paths partition file ownership. Each thread uses its own SSH pipes, append partial and source offset. Target size and complete SHA must pass before atomic rename and verified-byte increment. Individual transport failures retry with 15–120 second backoff. Integrity assertions stop the pool. Thread-local active file records and a mutex protect atomic state updates and unique completion counts. Historical retries persist across worker restarts. Small files and final full-manifest SHA remain after the large-file pool; retriever supplement remains serialized after primary acceptance.

Safe handoff: stop the old supervisor, identify its current task worker process group, stop only that group, verify old target receivers exited, preserve all partials, then restart one supervisor. No training process is touched. Original full source203 stays unchanged.

Validation: four simultaneous transfers with injected receiver disconnects, exact resumed output and all SHA equality; corrupt completed destination fails without overwrite. Independent review precedes actual handoff. Source and scripts are committed, pushed and remote SHA verified.
