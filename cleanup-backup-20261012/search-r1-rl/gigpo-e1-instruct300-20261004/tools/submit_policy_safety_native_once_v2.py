"""Single native-only submission. Preserve uncertain response; never submit again."""
import datetime,hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
EXPECTED_CANDIDATE_SHA256='7faa41598152201ca0a5e9cd1b41c06c6525ad1ffa2f8577360f29f929bd0a6a'
EXPECTED_PHYSICAL_SHA256='6f15809b7ff5d2eb0e17716cadb579e6c871d298b241b0a8843b4acb29ba1d2a'
CANDIDATE=ROOT/'logs/policy-safety-native-52befe9-candidate-v2.json'
REVIEW=ROOT/'logs/POLICY-SAFETY-NATIVE-52BEFE9-DEPLOYMENT-INDEPENDENT-REVIEW-v2.json'
PHYSICAL=ROOT/'logs/policy-safety-native-52befe9-physical-v2.json'
BINDING=ROOT/'logs/policy-safety-native-52befe9-reviewed-binding-v2.json'
INTENT=ROOT/'logs/policy-safety-native-52befe9-submit-intent-v2.json'
RECEIPT=ROOT/'logs/policy-safety-native-52befe9-submit-receipt-v2.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):
 with p.open('x') as f:
  f.write(json.dumps(value,indent=2)+'\n');f.flush();os.fsync(f.fileno())
 fd=os.open(str(p.parent),os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def main():
 assert not any(p.exists() for p in (BINDING,INTENT,RECEIPT)), 'Existing claim: inspect, never resubmit'
 assert sha(CANDIDATE)==EXPECTED_CANDIDATE_SHA256
 assert sha(PHYSICAL)==EXPECTED_PHYSICAL_SHA256
 c=json.loads(CANDIDATE.read_text());r=json.loads(REVIEW.read_text())
 assert r['status']=='PASS' and r['scope']=='GIGPO_POLICY_SAFETY_NATIVE_ONLY'
 assert r['candidate_sha256']==sha(CANDIDATE) and r['physical_sha256']==sha(PHYSICAL)
 assert r['submit_helper_sha256']==sha(Path(__file__)), 'Submission helper changed'
 assert c['status']=='CANDIDATE_NOT_READY' and c['code_commit']=='52befe968c856162d1bc79cd0171bb38309f4f9e'
 assert c['allowed_safety_modes']==['native_fixture'] and c['native_test_threshold_sets']==[[100.,1.]]
 assert c['helpers']==str(ROOT/'tools/policy-safety-native-52befe9-v2')
 assert c['resource_guard']['path']==c['helpers']+'/resource_guard_v2.py'
 assert c['resource_ownership']['path']==c['helpers']+'/resource_ownership_v2.py'
 assert c['resource_guard']['sha256']=='9644657c579eae63e5a70d0396deb7dfd93ddfd3f38a1604133982b6d195a444'
 assert c['resource_ownership']['sha256']=='7726925b1b02151a681cc1778b71b2c8984dfd87af01512179901d7faf9d8fed'
 assert c['restore_step']==25 and c['expected_optimizer_steps_per_outer']==10
 q=subprocess.check_output(['squeue','-h','-w','h100-8','-o','%i %j %T'],text=True)
 g=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True)
 assert not q.strip() and not g.strip(), 'Node is occupied: do not submit now'
 c.update(status='REVIEWED_FOR_NATIVE_ONLY',review_path=str(REVIEW),review_sha256=sha(REVIEW))
 c['evidence'] += [dict(path=str(REVIEW),sha256=sha(REVIEW)),dict(path=str(PHYSICAL),sha256=sha(PHYSICAL))]
 write(INTENT,dict(status='CLAIMED_NO_RESUBMISSION',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidate_sha256=sha(CANDIDATE),binding=str(BINDING),restore_checkpoint=c['restore_checkpoint'],actual_outer_updates=0,queue=q,gpu_compute=g))
 write(BINDING,c)
 done=subprocess.run(['sbatch','--parsable',str(Path(c['helpers'])/'native_safety_v1.sbatch'),str(BINDING),sha(BINDING),c['repo'],c['helpers']],text=True,capture_output=True)
 v=dict(stdout=done.stdout,stderr=done.stderr,returncode=done.returncode,binding=str(BINDING),binding_sha256=sha(BINDING),actual_training_submitted=False)
 job=done.stdout.strip().split(';')[0]
 if done.returncode==0 and job.isdigit():v['job']=job
 write(RECEIPT,v);print(json.dumps(v),flush=True)
 assert done.returncode==0 and job.isdigit(), 'Uncertain response: inspect receipt and scheduler; never repeat'
if __name__=='__main__':main()
