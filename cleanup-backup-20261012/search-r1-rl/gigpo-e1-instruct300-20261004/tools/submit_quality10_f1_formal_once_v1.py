"""Single durable submission claim; uncertain responses require inspection, never retry."""
import hashlib,json,os,subprocess,datetime,sys
from pathlib import Path
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
CANDIDATE=ROOT/'logs/quality10-formal-selected25-f1ed53b-candidate-v1.json'
REVIEW=ROOT/'logs/QUALITY10-F1-FORMAL-DEPLOYMENT-INDEPENDENT-REVIEW-v1.json'
BINDING=ROOT/'logs/quality10-formal-selected25-f1ed53b-reviewed-binding-v1.json'
INTENT=ROOT/'logs/quality10-f1-formal-submit-intent-v1.json'
RECEIPT=ROOT/'logs/quality10-f1-formal-submit-receipt-v1.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 with p.open('x') as f:f.write(json.dumps(v,indent=2)+'\n');f.flush();os.fsync(f.fileno())
 fd=os.open(str(p.parent),os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def main():
 assert not any(p.exists() for p in (BINDING,INTENT,RECEIPT)), 'Existing claim: inspect, never resubmit'
 assert not subprocess.check_output(['squeue','-h','-w','h100-8','-n','gigpo-e1-quality10-resume25','-o','%i %T'],text=True).strip(),'Live native exists'
 assert len(sys.argv)==3 and sha(CANDIDATE)==sys.argv[1] and sha(REVIEW)==sys.argv[2], 'Exact independently reviewed candidate/review SHA required'
 c=json.loads(CANDIDATE.read_text());review=json.loads(REVIEW.read_text())
 assert review['status']=='PASS' and review['candidate_sha256']==sha(CANDIDATE),'New independent deployment PASS required'
 assert c['status']=='CANDIDATE_NOT_READY' and c['code_commit']=='f1ed53b45d6f5d4d54e9fb2ef81994ebfa2117e3'
 assert c['restore_step']==25 and c['expected_optimizer_steps_per_outer']==10
 assert c['native_status']=='PASS' and c['native_job']=='3243', 'New actual native acceptance required'
 c.update(status='REVIEWED',review_path=str(REVIEW),review_sha256=sha(REVIEW))
 c['evidence'].append({'path':str(REVIEW),'sha256':sha(REVIEW)})
 write(INTENT,{'status':'CLAIMED_NO_RESUBMISSION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'candidate_sha256':sha(CANDIDATE),'binding':str(BINDING),'restore_checkpoint':c['restore_checkpoint'],'actual_outer_updates':0})
 write(BINDING,c)
 done=subprocess.run(['sbatch','--parsable','--job-name=gigpo-e1-quality10-resume25','--output='+str(ROOT/'logs/quality10-resume25-%j.log'),str(Path(c['helpers'])/'formal_quality10_v1.sbatch'),str(BINDING),sha(BINDING),c['repo'],c['helpers']],text=True,capture_output=True)
 raw={'stdout':done.stdout,'stderr':done.stderr,'returncode':done.returncode,'binding':str(BINDING),'binding_sha256':sha(BINDING)}
 job=done.stdout.strip().split(';')[0]
 if done.returncode==0 and job.isdigit():raw['job']=job
 write(RECEIPT,raw);print(json.dumps(raw),flush=True)
 assert done.returncode==0 and job.isdigit(),'Uncertain submission: preserve receipt, inspect scheduler, never repeat'
if __name__=='__main__':main()
