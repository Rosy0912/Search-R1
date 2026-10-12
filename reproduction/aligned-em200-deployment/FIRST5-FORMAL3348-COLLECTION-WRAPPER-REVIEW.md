# Formal3348 first5 collection wrapper review

Result: CODE_STATIC_ONLY; no must-fix found. Reviewer: alignment_generation_review. No remote action, collector execution, helper test rerun, release or job submission occurred.

Reviewed wrapper: requests/collect-first5-formal3348.py

SHA256: eba8e7c7ed4960c6b84fc41d32e2af0cf27a6630df4f2fe2372db506c0c38c34

Static AST/read-through verified fixed SOURCE080, formal3348, token66cb18840a404586ab3333f8c393b8b7 and exact canonical RUN. The canonical actual submission receipt must match these fields. Only a single original fifth-pause marker with the exact manifest path and no released marker permits the child invocation. Existing or symlinked output is refused. The helper SHA is fixed78d47e2653dd235f952c50b01cc1fc9baca525e11ecd17fa1a4e8202de9c8223; no helper/source/runtime bytes are changed.

The thin wrapper delegates the deeper nonce, same live driver PID/birth, manifestSHA, exact five attempts/20writes, native/EM/stage/config/source and actual receipt provenance checks to that already-reviewed helper. Read-through of the fixed helper confirms source080 clean checkout plus exact isolated Linux environment, canonical exclusive output creation, nonce format binding, manifest-member hashes, original same-PID pause and manifestSHA, and post-CPU input/live-prefix/process-birth checks. The pause marker itself has no nonce field; the actual manifest contains the nonce and its SHA binds it. This review does not repeat the previously signed helper tests or enlarge their acceptance scope.

Child invocation uses lexical newROOT/env/bin/python, cwd/PYTHONPATH newrepo, CUDA hidden, offlineHF/Transformers, OMP1, no bytecache and removes PYTHONOPTIMIZE. It calls only the CPU helper with a fresh ownROOT/evidence output path. Returned status is accepted only when the child exit is0 and cpu-evidence.json reports CPU_EVIDENCE_COLLECTED_NOT_INDEPENDENTLY_REVIEWED_NOT_RELEASE with an empty issues list. An incomplete helper can exit0, but the wrapper's final status check rejects it. Captured child output is explicitly labeled extraction, never independent acceptance/release. Timeout1700s, parse errors and unknown output fail rather than granting PASS; partial output must be preserved and inspected.

Limit: invocation prechecks are an inexpensive filter; actual correctness relies on the immutable helper's full checks. Concurrency with a hostile filesystem writer is outside the managed-directory assumption. Future actual fifth pause, CPU load success, independently reviewed evidence and release remain separate facts. The wrapper neither produces an independent PASS receipt nor releases training.
