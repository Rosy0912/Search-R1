"""Metadata freeze for runtime14 evidence-only actual job3325 repair."""
import hashlib,json
from pathlib import Path
B=Path.cwd();O=B/"outputs/gigpo-e1-instruct200";ROOT="/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006"
P=B/"work/gigpo-e1-quality-resume-runtime/safety-calibration-fresh200-v13-fsdpinit1"
N=P.with_name("safety-calibration-fresh200-v14-evidenceclone1")
R=ROOT+"/tools/fresh200-continuation-v14-evidenceclone1";OLD=ROOT+"/tools/fresh200-continuation-v13-fsdpinit1"
def sha(raw):return hashlib.sha256(raw).hexdigest()
def rec(p,path=None):
 raw=p.read_bytes();return dict(path=str(p) if path is None else path,bytes=len(raw),sha256=sha(raw))
def save(p,d):
 with p.open("x") as f:json.dump(d,f,sort_keys=True,indent=2);f.write("\n")
def dig(rows):return sha(json.dumps(rows,sort_keys=True,separators=(",",":")).encode())
reviewp=O/"RUNTIME14-EVIDENCECLONE1-INDEPENDENT-PREEXEC-REVIEW-v1.json";review=json.loads(reviewp.read_bytes())
if not(review["status"]=="PASS" and review["independent_review"] is True and review["blocking_findings"]==[]):raise RuntimeError("Missing genuine static review")
files=[];unchanged=[];changed=[]
for p in sorted(N.iterdir()):
 if not p.is_file() or p.is_symlink():raise RuntimeError("Unexpected runtime entry")
 row=dict(relative_path=p.name,bytes=p.stat().st_size,sha256=sha(p.read_bytes()));files.append(row)
 (unchanged if p.read_bytes()==(P/p.name).read_bytes() else changed).append(p.name)
if len(files)!=44 or changed!=["calibration_driver_v1.py"]:raise RuntimeError("Unexpected runtime changes")
parentdriver=(P/"calibration_driver_v1.py").read_bytes();childdriver=(N/"calibration_driver_v1.py").read_bytes()
if parentdriver.replace(b"v.detach().cpu() for k, v in batch.batch.items()",b"v.detach().cpu().clone() for k, v in batch.batch.items()",1)!=childdriver:raise RuntimeError("Beyond single clone")
manifestp=O/"FRESH200-RUNTIME14-EVIDENCECLONE1-STATIC-MANIFEST-v1.json"
authorp=O/"RUNTIME14-EVIDENCECLONE1-DERIVATION-v1.json"
manifest=dict(schema=1,status="STATIC_ONLY",runtime_version=14,runtime_variant="evidenceclone1",runtime_root=str(N),canonical_future_runtime_root=R,source_commit="2ddbb0a1619ce7cde28b4c5be110ffe5bd11c963",source_files_changed=False,files=files,file_count=44,unchanged_parent_files=unchanged,changed_files=changed,runtime_relative_files_digest=dig(files),runtime_parent_root=str(P),independent_static_review=rec(reviewp),author_derivation=rec(authorp),CPU_execution=False,GPU_execution=False,actual_probability_acceptance=None,narrow_repair="Clone only temporary rollout evidence tensor leaves into separate CPU storage; preserve active training tensors, all values/dtypes/mapping, algorithm/config and guards")
save(manifestp,manifest)
parentp=O/"FRESH200-STARTUP3323-REPAIR-BINDING-CANDIDATE-v1.json";b=json.loads(parentp.read_bytes())
newrows=[dict(path=R+"/"+r["relative_path"],bytes=r["bytes"],sha256=r["sha256"]) for r in files];newby={r["path"]:r for r in newrows}
b["files"]=[newby[R+row["path"][len(OLD):]] if row["path"].startswith(OLD+"/") else row for row in b["files"]]
b.update(helpers=R,runtime_files=newrows,runtime_version=14,runtime_variant="evidenceclone1",runtime_relative_files_digest=dig(files),runtime_canonical_files_digest=dig(sorted(newrows,key=lambda x:x["path"])),declared_primary_files_digest=dig(sorted(b["files"],key=lambda x:x["path"])),owned_submit_namespace=ROOT+"/logs/fresh200-evidence-repair3325-once-v1",submitted=False,actual_owned_job_id=None,actual_single_submit_review=None,actual_runtime_static_review=rec(reviewp,ROOT+"/logs/"+reviewp.name),actual_runtime_static_manifest=rec(manifestp,ROOT+"/logs/"+manifestp.name),binding_derivation="Unchanged scientific source/model/data/config; isolated44-file runtime14 changes only one temporary saved rollout-evidence leaf .clone. Parent3325 actually failed before probability approval or first Adam. No trained state or counted iteration discarded. No repeated CPU/control/native/test jobs. Queuing is not GPU admission; actual allocation/UUID-OFD/dual15/pidfd-cgroup checks retained.",startup_repair_parent_job="3325",startup_repair_parent_completed_outer=0,startup_repair_parent_actual_Adam_writes_per_rank=0)
terminalp=O/"job3325-terminal-originals-v1/TERMINAL-ORIGINALS-RECEIPT.json";terminal=json.loads(terminalp.read_bytes())
resultp=terminalp.parent/"run-result.json";result=json.loads(resultp.read_bytes())
drivert=json.loads((terminalp.parent/"continuation-terminal.json").read_bytes())
initial=json.loads((terminalp.parent/"actual-official-zero-initialization.json").read_bytes())
if result["status"]!="FAILED" or result["cleanup_residual_states"]!={} or terminal["recorded_cgroup_exists"] or any(terminal["owned_pid_exists"].values()) or terminal["audit_files"]!=[] or terminal["checkpoint_files"]!=[]:raise RuntimeError("Unknown or trained parent state")
if drivert["completed_outer_updates"]!=0 or drivert["worker_recovery_rpc_dispatched"] or drivert["preservation_errors"]!=[] or any(r["scheduler_step"]!=0 for r in initial["workers"]):raise RuntimeError("Parent no longer zero")
factp=O/"FRESH200-JOB3325-TERMINAL-FACTS-v1.json"
fact=dict(status="ACTUAL_DRIVER_FAILED_BEFORE_TRAINING",job_id="3325",completed_outer=0,actual_Adam_writes_per_rank=0,cleanup_residual_states={},owned_pid_exists=terminal["owned_pid_exists"],recorded_cgroup_exists=False,original_terminal_receipt=rec(terminalp),original_run_result=next(r for r in terminal["files"] if r["path"].endswith("/run-result.json")),original_initial_zero=next(r for r in terminal["files"] if r["path"].endswith("/actual-official-zero-initialization.json")),original_driver_terminal=next(r for r in terminal["files"] if r["path"].endswith("/continuation-terminal.json")),failure_site_before_probability_approval_and_first_Adam=True,scheduler_entry_expired=True,scheduler_exit_code=None,baseline_comparison_passed=True,baseline_f1=.2429115394461968,baseline_em=.18359375,failure_is_not_human_pause=True)
save(factp,fact);b["startup_repair_terminal_factual_record"]=rec(factp,ROOT+"/logs/"+factp.name)
extras=[(reviewp,ROOT+"/logs/"+reviewp.name),(manifestp,ROOT+"/logs/"+manifestp.name),(factp,ROOT+"/logs/"+factp.name),(authorp,ROOT+"/logs/"+authorp.name)]
for p,path in extras:b["evidence"].append(rec(p,path))
bindingp=O/"FRESH200-EVIDENCE-REPAIR3325-BINDING-CANDIDATE-v1.json";save(bindingp,b)
dp=O/"FRESH200-EVIDENCE-REPAIR3325-DERIVATION-v1.json"
save(dp,dict(parent_binding=rec(parentp),binding_candidate=rec(bindingp),runtime_manifest=rec(manifestp),runtime_review=rec(reviewp),parent_terminal_fact=rec(factp),extra_metadata_staging_local_to_remote=[dict(local_path=str(p),**rec(p,path)) for p,path in extras],new_tests_authorized=False,extra_GPU_validation_job_authorized=False,single_exclusive_training_submission=True,queued_submission_is_not_GPU_admission=True,actual_resource_checks_retained=True,first5_count_toward200=True,same_process_continuation_required=True))
print(json.dumps(dict(binding=rec(bindingp),derivation=rec(dp),runtime=rec(manifestp))))

