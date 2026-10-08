"""Exclusive byte-identical evidence alias. No CPU rerun/GPU/tool/source change."""
import argparse,datetime,hashlib,json,pathlib,re,os,time
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006')
SOURCE='a2568d0df306ead6e059760e28b5bd53ee3f6876'
LOG_SHA='ffafef808c5fee0e5f567efd8cb06d575f2e96f8a3d66e8d4437026b7e92395b'
EXIT_SHA='e9852d1760b3a79f0ac85fd552303bd318a12a8fa8a5fcadd08606f80fa9b5b3'
APPROVAL_SHA='400aedda99048c3c16229f26daa7a521b628fe83f7a774ebd981d5d435c6f0b2'
MANIFEST_SHA='8bd3fccd459f4b58a6dd048c624328b6fe4f067f8c26333aaef81d560e32ba46'
def need(v,msg):
 if not v:raise ValueError(msg)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def original(p):
 need(p.is_file()and not p.is_symlink()and p.resolve()==p,'Canonical ordinary original required '+str(p));raw=p.read_bytes();return raw

def create(job):
 start=time.monotonic();need(type(job)is int and job>0,'Actual positive hardware job required')
 logs=ROOT/'logs';need(logs.is_dir()and not logs.is_symlink()and logs.resolve()==logs,'Exact newROOT logs required')
 source=logs/'cpu-gate-durable.log';exitp=logs/'cpu-gate-durable-exit.json';target=logs/'cpu-gate.log';provenance=logs/'cpu-gate-log-compatibility-copy.json'
 need(not target.exists()and not target.is_symlink()and not provenance.exists()and not provenance.is_symlink(),'Exclusive new evidence only; never overwrite')
 need(original(ROOT/'source-expected.txt')==SOURCE.encode()+b'\n','Fixed new source marker required')
 raw=original(source);exitraw=original(exitp);need(sha(raw)==LOG_SHA and sha(exitraw)==EXIT_SHA,'Signed actual durable originals differ')
 exitrecord=json.loads(exitraw);need(type(exitrecord['exit_code'])is int and exitrecord['exit_code']==0 and exitrecord['source_commit']==SOURCE and exitrecord['log_sha256']==LOG_SHA and exitrecord['training_submitted']is False,'Actual durable exit/source differs')
 approvalraw=original(logs/'launch-approval.json');need(sha(approvalraw)==APPROVAL_SHA,'Signed launch approval differs');approval=json.loads(approvalraw)
 need(approval['status']=='PASS_INDEPENDENT_E1_LAUNCH_REVIEW'and approval['source_commit']==SOURCE and approval['root']==str(ROOT),'Signed newROOT approval differs')
 b=approval['cpu_gate_log'];need(b==dict(path=str(source),bytes=len(raw),sha256=LOG_SHA),'Approval primary log must remain actual durable')
 b=approval['cpu_gate_exit_record'];need(b==dict(path=str(exitp),bytes=len(exitraw),sha256=EXIT_SHA),'Approval original durable exit differs')
 need(sha(original(logs/'tools-manifest.json'))==MANIFEST_SHA and approval['tools_manifest_sha256']==MANIFEST_SHA,'Frozen36 tool manifest differs')
 run=ROOT/'runs'/f'bapo-instruct300-hardware-{job}';terminal=original(run/'slurm-terminal.txt');text=terminal.decode()
 for key,value in [('JobId',str(job)),('JobState','COMPLETED'),('ExitCode','0:0'),('DerivedExitCode','0:0'),('Restarts','0'),('NodeList','h100-5')]:need(re.findall(r'(?:^|\s)'+key+r'=([^\s]+)',text)==[value],'Actual completed terminal field differs '+key)
 inventory=original(run/'actual-fullhash-inventory.json');inv=json.loads(inventory);need(inv['job_id']==job and inv['source_commit']==SOURCE and inv['run_root']==str(run),'Actual sealed inventory identity differs')
 bound={x['path']:x for x in inv['evidence']};need(len(bound)==len(inv['evidence']),'Unique immutable original inventory')
 for p,r in ((source,raw),(exitp,exitraw),(logs/'launch-approval.json',approvalraw),(run/'slurm-terminal.txt',terminal)):
  need(bound[str(p)]['sha256']==sha(r)and bound[str(p)]['bytes']==len(r),'Original sealed binding differs '+str(p))
 need(str(target)not in bound and str(provenance)not in bound,'Compatibility evidence must be new post-terminal supplemental evidence')
 record=dict(status='POST_TERMINAL_BYTE_IDENTICAL_CPU_LOG_PATH_COMPATIBILITY_ONLY',root=str(ROOT),source_commit=SOURCE,hardware_job_id=job,original_cpu_run_preserved=True,cpu_rerun=False,gpu_execution=False,training_submitted=False,frozen_tools_modified=False,primary_log_path_in_approval=str(source),source_binding=dict(path=str(source),bytes=len(raw),sha256=LOG_SHA),original_exit_binding=dict(path=str(exitp),bytes=len(exitraw),sha256=EXIT_SHA),target_binding=dict(path=str(target),bytes=len(raw),sha256=LOG_SHA),signed_approval_sha256=APPROVAL_SHA,immutable_terminal_inventory_sha256=sha(inventory),terminal_sha256=sha(terminal),copy_tool_sha256=sha(pathlib.Path(__file__).read_bytes()),created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 with target.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 need(original(target)==raw and original(source)==raw and original(exitp)==exitraw,'Post-copy originals/target changed')
 record['copy_wall_seconds']=time.monotonic()-start
 with provenance.open('xb')as f:f.write((json.dumps(record,sort_keys=True,indent=2,allow_nan=False)+'\n').encode());f.flush();os.fsync(f.fileno())
 return dict(status=record['status'],target=dict(path=str(target),sha256=sha(original(target)),bytes=target.stat().st_size),provenance=dict(path=str(provenance),sha256=sha(original(provenance)),bytes=provenance.stat().st_size))
def main():
 p=argparse.ArgumentParser();p.add_argument('--job-id',type=int,required=True);a=p.parse_args();print(json.dumps(create(a.job_id)))
if __name__=='__main__':main()
