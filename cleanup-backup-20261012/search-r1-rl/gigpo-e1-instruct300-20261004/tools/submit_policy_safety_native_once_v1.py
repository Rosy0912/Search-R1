"""Single native-only submission. Preserve uncertain response; never submit again."""
import datetime,hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
CANDIDATE=ROOT/'logs/policy-safety-native-52befe9-candidate-v1.json'
REVIEW=ROOT/'logs/POLICY-SAFETY-NATIVE-52BEFE9-DEPLOYMENT-INDEPENDENT-REVIEW-v1.json'
PHYSICAL=ROOT/'logs/policy-safety-native-52befe9-physical-v1.json'
BINDING=ROOT/'logs/policy-safety-native-52befe9-reviewed-binding-v1.json'
INTENT=ROOT/'logs/policy-safety-native-52befe9-submit-intent-v1.json'
RECEIPT=ROOT/'logs/policy-safety-native-52befe9-submit-receipt-v1.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):
 with p.open('x') as f:
  f.write(json.dumps(value,indent=2)+'\n');f.flush();os.fsync(f.fileno())
 fd=os.open(str(p.parent),os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def main():
 assert not any(p.exists() for p in (BINDING,INTENT,RECEIPT)), 'Existing claim: inspect, never resubmit'
 assert sha(CANDIDATE)=='3276b298ee324e0d87719d0cf835af2ae2a93b22e4818dc65229decb3053c0d5'
 assert sha(PHYSICAL)=='c65db5f2df3be25b62d17945efe2a6091410c3bbcb935e036e86bbe64fd88099'
 c=json.loads(CANDIDATE.read_text());r=json.loads(REVIEW.read_text())
 assert r['status']=='PASS' and r['scope']=='GIGPO_POLICY_SAFETY_NATIVE_ONLY'
 assert r['candidate_sha256']==sha(CANDIDATE) and r['physical_sha256']==sha(PHYSICAL)
 assert r['submit_helper_sha256']==sha(Path(__file__)), 'Submission helper changed'
 assert c['status']=='CANDIDATE_NOT_READY' and c['code_commit']=='52befe968c856162d1bc79cd0171bb38309f4f9e'
 assert c['allowed_safety_modes']==['native_fixture'] and c['native_test_threshold_sets']==[[100.,1.]]
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
