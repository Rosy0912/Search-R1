# Stopped3315 node repair candidate: blocked pending controller diagnosis

Status: **NOT AUTHORIZED, NOT EXECUTED, NOT A PROVEN ROOT-CAUSE REPAIR.** UTC 2026-10-08T21:06:27.288394+00:00. Parent's later21:01Z root-cause instruction supersedes the earlier request to prepare DOWN/RESUME as the candidate remedy. No remote command, signal, node update, mapping change or job submission was performed here.

## Actual evidence changes the proposed action

Saved health6/node observation2 show h100-5 CPUAlloc0/AllocMem0/empty compute, old3315 COMPLETING and same unique new3345 PENDING with no RUN. Oldstop was explicit3315-only at2026-10-08T18:07:39.855509Z; it does not authorize node-global operations. Version21.08 official/installed manual ignores SIGTERM. The actual matched old daemon has PID755855/startticks1433986891/UID0/PPID1, Ubuntu slurmstepd-wlm SHAbe7770948fcfd24b074745d8add09005f77c1ae0ddb380b86abceb71d2bfdd6a, service cgroup, no children and no3315 job cgroups at the observed paths.

V2 actual observation21:01:48Z retains repeated exact3315.0 errors resolving `h100-16`, followed by unable to establish control-machine address/send job-complete message. This supports a completion RPC/address problem, rather than proof that CPU/GPU work is still running. The old controller hostname was reported retired; its replacement/current controller identity has not been independently bound by this review. Service cgroup membership is not a job-specific ownership certificate. No task/cgroup/spool deletion is justified.

DOWN is node-global, may terminate other allocations and does not fix old-daemon DNS/controller routing. RESUME can schedule any queued job and does not certify old cleanup. Even an eventual correct node-reconciliation decision requires additional explicit human authorization, fresh foreign-allocation refusal and separate receipts. An old stop or current BAPO goal is not such authorization.

## Concrete next step: one bounded read-only controller comparison

Before choosing a mutation, root may publish/review a dedicated read-only request, then retain its original exit/stdout/stderr and exact SHA/time/host/boot:

1. Bind actual h100-5, installed scontrol21.08.5 and the exact old daemon PID/startticks/exeSHA before and after inspection. Read only selected old process environment keys `SLURM_CONF` and `SLURM_CONF_SERVER` (record absent honestly); never dump full environment or credentials. Read selected current CLI environment keys separately. Record lexical/resolved config paths, file inode/size/mtime/SHA, without printing complete configuration.
2. Run `scontrol ping` and `scontrol show config` read-only. Keep only controller host/address/port/config identity fields (`SlurmctldHost`, `ControlMachine`, `ControlAddr`, `BackupController`, `BackupAddr`, `SlurmctldPort`, `SlurmdPort`, relevant configuration path). For a pinned readable local config, extract only the same selected directives and bounded Include path metadata. Do not alter controller/daemon config, global environment or debug flags. State explicitly that an on-disk file is not proof of the old daemon's cached in-memory config.
3. Resolve exact old `h100-16` and controller names present in those selected fields with bounded read-only `getent ahostsv4`/resolver queries. Retain exit/stderr/addresses and hostname-to-IP provenance; do not guess the new controller IP from node names or unrelated receipts, and do not update /etc/hosts/DNS/global config.
4. Compare successful current CLI controller endpoint with exact old-daemon/config hostname and current controller's known3315 identity. Retain filtered3315 completion/error log rows and hashes already collected, plus selected config/resolution observations. A mapping or alias to an unrelated controller would be unsafe; the same controller lineage/job identity must be established before proposing any mapping repair.
5. Produce a diagnosis table: current CLI endpoint/resolution/reachability; old selected config path/host provenance; actual repeated completion target; evidence missing; possible minimal administrator repair. If old in-memory target still needs a retired name, an administrator may need compatible name/address reconciliation or completion-state reconciliation. This remains a conditional proposal, not a repair command or approval.

No controller/slurmd restart, SIGTERM/KILL, cgroup/spool removal, reboot, Slurm debug change, node update, new3345 submission, resume3315 or bypass of Slurm belongs in this read-only request. It must not broaden task scope or touch other experiments. Missing observations stop diagnosis; they do not automatically escalate.

## Retained, hard-disabled earlier candidate

`node_completion_repair.py` contains the earlier narrow review draft and local plan command, but `NODE_RECONCILIATION_ADMISSION=False` causes apply to reject before authorization I/O, evidence creation or any subprocess. `plan` records execution_admission=false and BLOCKED_OLD_H10016_CONTROLLER_COMPLETION_RPC_DNS_UNRESOLVED. No authorization file has been manufactured. Do not deploy or enable this draft as a remedy. It is retained as an auditable superseded design, not validated execution code.

The earlier design describes fresh exact node/old3315/new3345/allnode allocations/emptyGPU/listpids/ghostbirth checks, old logs/source archival, exclusive plan/auth/intent/per-command receipt, DOWN followed by exact node/own-job checks and then RESUME. Unknown/nonzero/preflight failures stop without retry/escalation, potentially leaving the node DOWN; that operational consequence must be included in any future approval. There are no daemon/cgroup/spool operations or second sbatch. Its runtime path has not been exercised and has no independent execution review. In particular, original daemon disappearance/oldjob retirement after DOWN would need a fresh reviewed policy, not ad-hoc escalation.

Local **11 boundary tests PASS0.02s** use saved historical observations, not fresh admission. They reject foreign allocated jobs/CPU/memory/GPU, changed process birth, duplicate newjob and started3345; verify hard-blocked apply invokes no runner and creates no output; verify exclusive plan/receipt writes. These tests do not prove DOWN/RESUME effectiveness or authorization. Tests require the named local saved receipts and are not a target deployment suite.

## Frozen saved receipts

| File | SHA256 |
| --- | --- |
| aligned-em200-hardware3345-health6.actual.json | `e4e3e1bd1696be9517518c6fc224d4b675ed7f21b997018bcdac2f5849ff762f` |
| aligned-em200-node-queue3345-observation2.actual.json | `fca3a1177fdf0461ecc35afe375b72b28b49b9d25b5c3f8ca09ab435af0694e4` |
| aligned-em200-old3315-completion-readonly-observation2.actual.json | `e128229263c19537cc4722284be31d9aaef7a30e167937508fa010ebaaf5d9c2` |
| aligned-em200-old3315-stepdaemon-diagnosis.actual.json | `5961713de438e50f17abea7118e843b34d41ac36d80ecbc585777d79bd8f3979` |
| aligned-em200-old3315-stepdaemon-missing-logs.actual.json | `f6acf880d6269b79e3504cca3bd9cf5ee59d9944cfa51138c5c6e3021d870397` |

## Local candidate bytes

| File | SHA256 |
| --- | --- |
| node_completion_repair.py | `7ec3eec716ff6a8319a7916a6044a3c7a5e1c375d269b1646b2f8162fc320558` |
| test_node_completion_repair.py | `f17df3397de74d6ebedf7ee3e3d5f679c2d9041fca1089c3c28dacac8593ee43` |
