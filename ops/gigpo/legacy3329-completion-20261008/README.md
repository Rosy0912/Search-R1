# Scoped legacy Slurm completion compatibility

Session: 01a0fcbb-863f-7450-926d-db537f648265.
Scientific source is unchanged: 2ddbb0a1619ce7cde28b4c5be110ffe5bd11c963.
Parent full implementation backup: 703601c1a42c9d0f384875a64d031e77b93a0ccf.

Human authorized stopping the existing Search-G1 job3329 on Wuhan H100 h100-8 and starting the existing, exactly-once GiGPO job3332. Actual GPU tasks and original cgroup are gone; the remaining original3329.batch stepd has a cached retired h100-16 controller name. Current controller is h100-8. The repair lets that original stepd naturally deliver its completion RPC.

The helper verifies both original job identities, human authorization, cancellation intent, frozen3332 binding, no live tasks, current local controller, and fresh original-stepd DNS errors. It only appends its unique temporary resolver line; cleanup comments its own prefix without rewriting hosts. A detached, acknowledged watchdog restores resolution semantics within a bounded interval. It does not fabricate exit status, force node state, restart Slurm, submit jobs, or change model/training code.

This is an operations artifact, not a training acceptance record. Do not replay the transport. Existing intent/receipt must be read after uncertain execution. All SSH layers retain strict host checking and use the confirmed retained18 route. The commented resolver line is semantic restoration, not byte-for-byte original restoration. Advisory file locking cannot prevent unrelated non-cooperative in-place writers; detected unknown changes fail explicitly.

An actual clean-context nonauthor static review is required and bound to both script hashes. Execution results and job progress are recorded separately under the current session outputs and run report.
