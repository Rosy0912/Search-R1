"""Single durable submission claim; uncertain responses require inspection, never retry."""
import hashlib,json,os,subprocess,datetime
from pathlib import Path
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
CANDIDATE=ROOT/'logs/quality10-native-selected25-f1ed53b-candidate-v1.json'
REVIEW=ROOT/'logs/QUALITY10-F1-NATIVE-DEPLOYMENT-INDEPENDENT-REVIEW-v1.json'
BINDING=ROOT/'logs/quality10-native-selected25-f1ed53b-reviewed-binding-v1.json'
INTENT=ROOT/'logs/quality10-f1-native-submit-intent-v1.json'
RECEIPT=ROOT/'logs/quality10-f1-native-submit-receipt-v1.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 with p.open('x') as f:f.write(json.dumps(v,indent=2)+'\n');f.flush();os.fsync(f.fileno())
 fd=os.open(str(p.parent),os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def main():
 assert not any(p.exists() for p in (BINDING,INTENT,RECEIPT)), 'Existing claim: inspect, never resubmit'
 assert not subprocess.check_output(['squeue','-h','-w','h100-8','-n','gigpo-e1-quality10-native','-o','%i %T'],text=True).strip(),'Live native exists'
 assert sha(CANDIDATE)=='72750e4f17d7f0b4c68b3f79c2b18d37d5810803694602b84bfb5773034d7154'
 c=json.loads(CANDIDATE.read_text());review=json.loads(REVIEW.read_text())
 assert review['status']=='PASS' and review['candidate_sha256']==sha(CANDIDATE),'New independent deployment PASS required'
 assert c['status']=='CANDIDATE_NOT_READY' and c['code_commit']=='f1ed53b45d6f5d4d54e9fb2ef81994ebfa2117e3'
 assert c['restore_step']==25 and c['expected_optimizer_steps_per_outer']==10
 c.update(status='REVIEWED_FOR_NATIVE_ONLY',review_path=str(REVIEW),review_sha256=sha(REVIEW))
 c['evidence'].append({'path':str(REVIEW),'sha256':sha(REVIEW)})
 write(INTENT,{'status':'CLAIMED_NO_RESUBMISSION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'candidate_sha256':sha(CANDIDATE),'binding':str(BINDING),'restore_checkpoint':c['restore_checkpoint'],'actual_outer_updates':0})
 write(BINDING,c)
 done=subprocess.run(['sbatch','--parsable',str(Path(c['helpers'])/'native_quality10_v1.sbatch'),str(BINDING),sha(BINDING),c['repo'],c['helpers']],text=True,capture_output=True)
 raw={'stdout':done.stdout,'stderr':done.stderr,'returncode':done.returncode,'binding':str(BINDING),'binding_sha256':sha(BINDING)}
 job=done.stdout.strip().split(';')[0]
 if done.returncode==0 and job.isdigit():raw['job']=job
 write(RECEIPT,raw);print(json.dumps(raw),flush=True)
 assert done.returncode==0 and job.isdigit(),'Uncertain submission: preserve receipt, inspect scheduler, never repeat'
if __name__=='__main__':main()
