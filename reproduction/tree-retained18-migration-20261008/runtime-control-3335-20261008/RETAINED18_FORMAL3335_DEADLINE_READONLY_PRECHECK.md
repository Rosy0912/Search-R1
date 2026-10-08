# 3335 read-only deadline precheck

Current168h allocation can expire before final evaluation/delivery. The frozen launch selected168h; gpu partition MaxTime is UNLIMITED. With pre-dev outer proxy47.93min, dev512 projection15.43min, final51713 projection25.99h, the full196 new outer +40dev +final scenario is about193h, about25h beyond168h. Full outer step wall has not been published. This is a scenario, not promised ETA, measured final duration or completion.

Completed: 2026-10-08T05:43:29.848028+00:00 / 2026-10-08T13:43:29.848028+08:00
Scheduler observed: 2026-10-08T05:36:42.590589+00:00 / 2026-10-08T13:36:42.590589+08:00
Identity: helperh100-10/10.33.0.18; hostnameh100-10; uid0root. Clusterwuhan-h100 comes from frozenmanifest; global config not queried.

3335 RUNNING. Start2026-10-08 12:31:23 Beijing; TimeLimit7-00:00:00; End2026-10-15 12:31:23 Beijing/04:31:23UTC. Remaining166.911h at query. Deadline=N/A does not removeEndTime. Restarts0/Requeue0/SuspendTimeNone, NodeList/BatchHosth100-10, QOS/Accountnull.
PartitiongpuUP: MaxTimeUNLIMITED, DefaultTimeNONE, QoSN/A, AllowQosALL, DisableRootJobsNO. No finite partitioncap observed; other controller policy layers not audited. No update submitted.

Provenance: package launch.sbatch has no time directive and explicitly delegates resources/time to renderer. Renderer requires positive finite explicit wall_hours. Manifestdispatch.train_wall_hours168; actualsbatch --time168:00:00 and SHA matchesdeployment. Currenthuman-authority specifies goal200/final51713; no explicit7day humanhardbudget found in inspected records. Original raw goal message unavailable in this subtask: root must preserve independently known human hard budgets.

Startup pin: slurm_identity109-142 exactTimeLimit168h. launch/supervisor69/guard_entry49 verify_runtime at startup. Periodic guarded_run/NVML wrappers only verify_runtime_cgroup. Pinned resource_guard_retained18 has no liveTimeLimit/scontrol read. owned_lease_identity checks originalsavedallocation wall_hours168/SHA, no liveTimeLimit. Externaltree_shared_owned_runtime_fixed SHA4288a4dcfa923df4d8d74f009bf4c56c8446d04fa8ef4202c5e55d3371539ba0 matchesmanifest; no TimeLimit/schedulerdeadline logic. Preserve original168h identity and separate live extension record. Do not editmanifest/hotcode or rerunstartup/restartguard.

Protocol: actuallyloadedparent4;3335needs196newouter. Completed5(segmentnew1), remaining195fromthatboundary. dev512every5=>40dev; HFsave5/50/100/150/200; final51713. Keep200 and allturn/token/length/update budgets. No addedbeforetrain evaluation or5stepapprovalhold.

Rollout5 only1407.234s, not completeouter. Actualrolloutstart-to-validation_start proxy2875.741s/47.929min excludes currentdev/HFsave; publishedtiming_s/step followsdev/HFsave and was absent at metadata read.
dev5start 2026-10-08T13:23:47.399592+08:00; metadata 2026-10-08T13:37:47.774550+08:00. Entered15/16batches(480rows); no validation_completed/report/score. Entered is not complete/scored.14batch transitions mean57.863s=>dev16batches15.430min; final1617batches25.990h. Futuremodel/dataset cost may change. HF5oldmetadata13:32:57absent; notrefreshedhere; rolling5isnotHFsave/reload proof.

Formula fromstart:startup+196*predevouter+40*dev512+final51713+unknownextraHFsave/reload/delivery. Fromcompleted5:195outer+currentdevremainder+39futuredev+final. Publishedstep includesdev; subtractdev/save before comparing to proxy to avoid doublecount.

| Pre-dev outer | dev512 | final51713 | Total excluding extra saves/delivery | Over168h | Margin240h |
|---|---|---|---|---|---|
| 47.93min | 15.43min | 25.99h | 192.92h | +24.92h | +47.08h |
| 45.00min | 15.43min | 25.99h | 183.35h | +15.35h | +56.65h |
| 40.00min | 15.43min | 25.99h | 167.02h | -0.98h | +72.98h |
| 35.00min | 15.43min | 25.99h | 150.68h | -17.32h | +89.32h |
| 60.00min | 20.00min | 33.69h | 243.10h | +75.10h | -3.10h |

At same dev/final and noextraoverhead, averagepredevouter must be <=40.30min to fit168h. Observedproxy train+periodic endsaround2026-10-15T11:27:09.095637+08:00, leavingroughly1h for roughly26hfinal. Extrasaves could push cutoff beforeouter200. Scenario only.

For root decision only; NOT EXECUTED: totalTimeLimit240h/10days on sameRUNNING3335, targetEnd2026-10-18 12:31:23Beijing/04:31:23UTC. Partition has nofinitecap; rootprivilegedidentity observed. [OfficialSlurm scontrol docs](https://slurm.schedmd.com/scontrol.html) permit privileged increases of runningTimeLimit. Actualcontroller acceptance requires exactreturn/readback.
Proposed text only: scontrol --local update JobId=3335 TimeLimit=10-00:00:00. Preserve immutable168h startup pins; separate livebefore/after record.240h gives47hmargin for observedproxy;60minouter/20mindev sensitivity243h slightly exceeds240h. Recheck futureactualcadence. Largerfinite14days336h also below noobservedpartitioncap if root decides; no completion guarantee.

Required change records:
- Preserve original frozen168h manifest/sbatch/allocation-identity.json and SHAs as startup facts; record extension separately, do not rewrite pins.
- Record valid human scope/budget decision, UTC/BJ, hostname/IP/uid, source/manifest/config/run, exact before StartTime/TimeLimit/EndTime/NodeList/state/Restarts/Requeue.
- Retain exact update argv/returncode/stdout/stderr; proposed or failed request is not success.
- Readback only own3335: actual240h(or chosen value), EndTime, identicalStartTime/job/run/node/Restarts0/Requeue0, continuedRUNNING; separately retain fresh own resource sample.
- Do not rerun startup verify_runtime or restart guard/trainer. Keep saved168h identity and source pins.
- Keep200outer and all token/length/turn/update budgets, dev512every5, save5/50/100/150/200 and final51713. No hotcode/environment/manifest edits.
- Preserve source backup remoteSHA proofs and final actual evaluation/reload/cost records; extension is no completion or backup proof.

Only localdeadlineJSON/md written. No scheduler/training/environment change, tests, tensor/model load, GPU work, signals, suspend/kill/restart/requeue/submit, otherjob/queue query. Training does not wait. JSON retains fullremote command/return/stdout/stderr/UTC/BJ/identity and localsource hashes.

Actual read-only command receipts:

```text
hostname
returncode=0
h100-10
stderr=''
```

```text
id
returncode=0
uid=0(root) gid=0(root) groups=0(root),999(docker)
stderr=''
```

```text
date -u --iso-8601=seconds
returncode=0
2026-10-08T05:36:42+00:00
stderr=''
```

```text
date --iso-8601=seconds
returncode=0
2026-10-08T13:36:42+08:00
stderr=''
```

```text
scontrol --local show partition gpu
returncode=0
PartitionName=gpu
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=YES QoS=N/A
   DefaultTime=NONE DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
   MaxNodes=UNLIMITED MaxTime=UNLIMITED MinNodes=0 LLN=NO MaxCPUsPerNode=UNLIMITED
   Nodes=h100-[8,5,10-11]
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=NO
   OverTimeLimit=NONE PreemptMode=OFF
   State=UP TotalCPUs=768 TotalNodes=4 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerNode=UNLIMITED MaxMemPerNode=UNLIMITED
stderr=''
```

```text
scontrol --local show job 3335
returncode=0
JobId=3335 JobName=tree-fresh200-train
   UserId=root(0) GroupId=root(0) MCS_label=N/A
   Priority=4294898731 Nice=0 Account=(null) QOS=(null)
   JobState=RUNNING Reason=None Dependency=(null)
   Requeue=0 Restarts=0 BatchFlag=1 Reboot=0 ExitCode=0:0
   RunTime=01:05:19 TimeLimit=7-00:00:00 TimeMin=N/A
   SubmitTime=2026-10-08T12:31:22 EligibleTime=2026-10-08T12:31:22
   AccrueTime=2026-10-08T12:31:22
   StartTime=2026-10-08T12:31:23 EndTime=2026-10-15T12:31:23 Deadline=N/A
   SuspendTime=None SecsPreSuspend=0 LastSchedEval=2026-10-08T12:31:23 Scheduler=Main
   Partition=gpu AllocNode:Sid=h100-10:138026
   ReqNodeList=h100-10 ExcNodeList=(null)
   NodeList=h100-10
   BatchHost=h100-10
   NumNodes=1 NumCPUs=192 NumTasks=1 CPUs/Task=32 ReqB:S:C:T=0:0:*:*
   TRES=cpu=192,mem=256G,node=1,billing=192
   Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=*
   MinCPUsNode=32 MinMemoryNode=256G MinTmpDiskNode=0
   Features=(null) DelayBoot=00:00:00
   OverSubscribe=NO Contiguous=0 Licenses=(null) Network=(null)
   Command=(null)
   WorkDir=/root
   StdErr=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/node-move-logs/3335.err
   StdIn=/dev/null
   StdOut=/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/node-move-logs/3335.out
   Power=
   TresPerNode=gres:gpu:8
stderr=''
```
