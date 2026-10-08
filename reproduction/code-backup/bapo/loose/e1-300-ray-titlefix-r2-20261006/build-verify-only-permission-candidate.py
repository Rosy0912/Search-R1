"""Build an UNSIGNED verify-only candidate from actual reviewed allocation originals."""
import argparse,hashlib,json,pathlib
B=pathlib.Path(__file__).parent
ROOT='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-titlefix-r2-20261006';SOURCE='3f85dabc7aa05bc45ec7af015d65d9fa4a4c17e4'
def need(v,msg):
 if not v:raise ValueError(msg)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def build(capture,check,job):
 d=json.loads(pathlib.Path(capture).read_text());r=json.loads(pathlib.Path(check).read_text());run=ROOT+'/runs/bapo-instruct300-hardware-'+str(job)
 need(type(job)is int and job>0 and r['status']=='PASS_INDEPENDENT_ALLOCATED_INPUT_CHECK_NOT_SIGNED_PERMISSION_OR_GPU'and r['job_id']==job and r['source_commit']==SOURCE and r['run_root']==run,'Only this actual checked hardware allocation allowed')
 need(r['allocation_capture_sha256']==sha(capture)and d['job_id']==job and d['source_commit']==SOURCE and d['run_root']==run,'Original allocation capture binding differs')
 need(r['actual_gpu_accepted']is False and r['permission_signed']is False and r['training_authorized']is False,'No prior actual GPU/training permission allowed')
 need(r['manifest_sha256']==d['files']['recovery-manifest.json']['sha256'],'Actual new manifest binding differs')
 evidence=[]
 for group,names in [('files',('launch-approval.json','tools-manifest.json','cpu-gate.json')),('root_inputs',('final-launch-independent-review.md','runtime-executable.json','cpu-gate-durable.log','cpu-gate-durable-exit.json','title-dependency-preflight.json','title-dependency-preflight.log','title-dependency-preflight-exit.json','guardian-tests-durable.log','guardian-tests-durable-exit.json'))]:
  for name in names:
   x=d[group][name];evidence.append({k:x[k]for k in ('path','sha256','bytes')})
 return dict(status='CANDIDATE_REQUIRES_INDEPENDENT_BAPO_RESUME25_GPU_VERIFY_PERMISSION',scope='UNSIGNED_ACTUAL_HARDWARE_ALLOCATION_CANDIDATE_NO_TRAINING_OR_GPU_ACCEPTANCE',target_source_commit=SOURCE,target_run_root=run,target_job_id=job,manifest_sha256=r['manifest_sha256'],sampling_continuity='LEGACY25_GENERATION_RNG_RECONSTRUCTED_NOT_BITWISE_EXACT',independent_review=dict(path=run+'/HARDWARE'+str(job)+'-VERIFY-ONLY-INDEPENDENT-REVIEW.md',sha256=None,bytes=None,pending_independent_review=True),code_review_evidence=evidence,allocation_raw_sha256=sha(capture),allocated_input_check_sha256=sha(check),actual_gpu_accepted=False,training_authorized=False,permission_signed=False,production_training_permission_signed=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--capture',required=True);p.add_argument('--check',required=True);p.add_argument('--job-id',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args();out=pathlib.Path(a.output);need(not out.exists(),'Fresh unsigned output required');need(pathlib.Path(ROOT)not in out.resolve().parents,'Never write production permission');r=build(a.capture,a.check,a.job_id)
 with out.open('x')as f:f.write(json.dumps(r,indent=2)+'\n')
 print(json.dumps(dict(status='UNSIGNED_ONLY_NOT_PERMISSION',path=str(out.resolve()),sha256=sha(out))))
if __name__=='__main__':main()
