"""Local preparation only: bind completed independent hardware receipt to reviewed launcher."""
import argparse,base64,datetime,hashlib,json
from pathlib import Path
ROOT='/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009'
SOURCE='080ebbc46b89d61567cd33e1a5e2f1b40afd8b84'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--review',type=Path,required=True);p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 raw=a.review.read_bytes();review=json.loads(raw)
 if (review.get('status')!='PASS_INDEPENDENT_ALIGNED_HARDWARE_REVIEW' or review.get('scope')!='ACTUAL_8RANK_SYNTHETIC_AND_LIVE_RETRIEVAL_NOT_TRAINING' or review.get('source')!=SOURCE or review.get('root')!=ROOT or str(review.get('job'))!='3345'):raise RuntimeError('Actual independently accepted hardware3345 receipt required')
 here=Path(__file__).resolve().parent;files=json.loads(a.inventory.read_text())['files']
 for n,h in files.items():
  if sha((here/n).read_bytes())!=h:raise RuntimeError('Reviewed formal candidate changed: '+n)
 auth={'status':'AUTHORIZED_SINGLE_FRESH200_ALIGNED_LAUNCH','scope':'EXPLICIT_FORMAL_LAUNCH_AUTHORIZATION_NOT_COMPLETED_TRAINING','source':SOURCE,'root':ROOT,'hardware_review_sha256':sha(raw),'same_process_first5_gate':True,'authority':'Existing human-authorized baseline-aligned fresh200 execution goal; does not request new authorization or reuse old3315','configuration':'Full mini128/20writes, reference KL.001, ordinaryPPO, 57outerwarmup,4tool+finalnosearch,perouter initial/any5/retained EM; retain BAPO reward/stage mechanisms','prepared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 authraw=(json.dumps(auth,sort_keys=True,indent=2)+'\n').encode()
 old=(here/'formal_admission.py').read_text()
 if 'HARDWARE_REVIEW_SHA256 = None' not in old or 'FORMAL_AUTHORIZATION_SHA256 = None' not in old or 'FORMAL_FILES = {}' not in old:raise RuntimeError('Candidate must be previously unbound')
 new=old.replace('HARDWARE_REVIEW_SHA256 = None','HARDWARE_REVIEW_SHA256 = '+repr(sha(raw))).replace('FORMAL_AUTHORIZATION_SHA256 = None','FORMAL_AUTHORIZATION_SHA256 = '+repr(sha(authraw))).replace('FORMAL_FILES = {}','FORMAL_FILES = '+repr(files))
 compile(new,'formal_admission.py','exec')
 a.output.mkdir(exist_ok=False)
 blobs={'hardware-independent-acceptance.json':raw,'formal-launch-authorization.json':authraw,'formal_admission.py':new.encode()}
 for n,b in blobs.items():(a.output/n).write_bytes(b)
 plan={'source':SOURCE,'root':ROOT,'hardware_sha256':sha(raw),'authorization_sha256':sha(authraw),'admission_sha256':sha(new.encode()),'files':files,'scope':'ACTUAL_REVIEW_BOUND_PREPARATION_NOT_SUBMITTED'}
 (a.output/'binding.json').write_text(json.dumps(plan,indent=2)+'\n')
 header="from pathlib import Path\nimport base64,hashlib,json,os,subprocess\nR=Path("+repr(ROOT)+")\nif os.uname().nodename!='h100-5' or R.resolve()!=R:raise RuntimeError('Fixed canonical node required')\nif subprocess.check_output(['git','-C',str(R/'repo'),'rev-parse','HEAD'],text=True).strip()!="+repr(SOURCE)+":raise RuntimeError('Source changed')\nif (R/'scheduler-head/formal-submit-intent.json').exists():raise RuntimeError('Already submitted; no install')\n"
 write="""
def install(p,b,mode=0o644):
 if p.parent.resolve()!=p.parent or p.is_symlink():raise RuntimeError('Canonical own file required')
 if p.exists():
  if not p.is_file() or p.read_bytes()!=b:raise RuntimeError('Existing different file; refuse overwrite '+str(p))
 else:
  with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 os.chmod(p,mode)
 print(json.dumps({'file':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
"""
 part1=header+write
 for n in ('hardware-independent-acceptance.json','formal-launch-authorization.json'):
  part1+='install(R/"logs"/'+repr(n)+',base64.b64decode('+repr(base64.b64encode(blobs[n]).decode())+'))\n'
 part2=header+write+"C=R/'tools/deployment/candidate-formal-v1'\n"
 for n,h in files.items():
  part2+='b=(C/'+repr(n)+').read_bytes()\nif hashlib.sha256(b).hexdigest()!='+repr(h)+":raise RuntimeError('Candidate bytes changed')\ninstall(R/'tools/deployment'/"+repr(n)+",b,"+('0o755' if n.endswith(('.sh','.sbatch')) else '0o644')+')\n'
 n='submit_formal.py';h=sha((here/n).read_bytes());part2+='b=(C/'+repr(n)+').read_bytes()\nif hashlib.sha256(b).hexdigest()!='+repr(h)+":raise RuntimeError('Submit candidate changed')\ninstall(R/'tools/deployment'/"+repr(n)+",b)\n"
 part2+='install(R/"tools/deployment/formal_admission.py",base64.b64decode('+repr(base64.b64encode(blobs['formal_admission.py']).decode())+'))\n'
 for i,t in enumerate((part1,part2),1):
  compile(t,'install-bound-formal-part'+str(i)+'.py','exec');(a.output/('install-bound-formal-part'+str(i)+'.py')).write_text(t)
 print(json.dumps(plan))
if __name__=='__main__':main()
