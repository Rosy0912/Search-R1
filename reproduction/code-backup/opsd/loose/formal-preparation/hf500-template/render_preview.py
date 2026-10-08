"""Only local previews; never mutate source, deployed stages, or submit."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'deployment-bundles/hf-runtime-8e60976d364a-pilot-2994-reviewed-47cab4b02661'
FORMAL=ROOT/'deployment-bundles/formal-8e60976d364a-pilot-2994-hf-2996-reviewed-fd4b1b34cf20'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def once(text,old,new):assert text.count(old)==1,old;return text.replace(old,new,1)
def render(out):
 out=Path(out);assert not out.exists();out.mkdir(parents=True)
 original=json.loads((OLD/'source-manifest.json').read_text());formal=json.loads((FORMAL/'source-manifest.json').read_text())
 for folder,m,key in ((OLD,original,'file_sources'),(FORMAL,formal,'formal_sources')):
  for name,v in m[key].items():assert sha(folder/name)==v
 name='hf500-runtime-8e60976d364a-formal-2999';old=original['directory_name']
 wrapper=(OLD/'reload_job.sh').read_text().replace(old,name)
 wrapper=once(wrapper,'set -euo pipefail','set -euo pipefail\necho "STAGED PREVIEW: independent review required" >&2\nexit 78')
 wrapper=once(wrapper,': "${SLURM_JOB_ID:?Must run in own allocation}"',': "${SLURM_JOB_ID:?Must run in own allocation}"\n: "${OPSD_FORMAL_COMPLETION_RECEIPT:?Actual successful formal2999 completion receipt required}"')
 wrapper=once(wrapper,'opsd-new-hf-reload-$SLURM_JOB_ID','opsd-formal-hf-reload-$SLURM_JOB_ID')
 wrapper=once(wrapper,'opsd-new-acceptance-2994/output/checkpoints/opsd-new-gpu-acceptance-5/actor/global_step_5','opsd-formal-2999/output/checkpoints/opsd-gpu-formal500/actor/global_step_500')
 before='[[ -f "$root/runs/opsd-new-acceptance-2994/training-finished.txt" ]]\n[[ $(cat "$root/runs/opsd-new-acceptance-2994/exit.txt") == 0 ]]'
 wrapper=once(wrapper,before,'python "$root/tools/'+name+'/formal_completion_check.py"')
 wrapper=wrapper.replace('job_2994','job_2999').replace("previous_training_job='2994'","previous_training_job='2999'")
 (out/'reload_job.sh').write_text(wrapper)
 for filename in ('checkpoint_reload.py','resource_sample.py'):(out/filename).write_bytes((OLD/filename).read_bytes())
 (out/'training_evidence.py').write_bytes((FORMAL/'verify_formal_prerequisites.py').read_bytes())
 (out/'formal_completion_check.py').write_bytes((HERE/'formal_completion_check.py').read_bytes())
 (out/'periodic_validation.py').write_bytes((HERE/'periodic_validation.py').read_bytes())
 checker=(OLD/'source_bundle_check.py').read_text()
 checker=once(checker,"assert manifest['stage']=='hf_only_actual_pilot_no_future_reload'","assert manifest['stage']=='hf500_only_actual_formal_no_future_reload'")
 checker=once(checker,"{'reload_job.sh','checkpoint_reload.py','resource_sample.py','source_bundle_check.py'}","{'reload_job.sh','checkpoint_reload.py','resource_sample.py','source_bundle_check.py','formal_completion_check.py','training_evidence.py','periodic_validation.py'}")
 (out/'source_bundle_check.py').write_text(checker)
 files=('reload_job.sh','checkpoint_reload.py','resource_sample.py','source_bundle_check.py','formal_completion_check.py','training_evidence.py','periodic_validation.py')
 m={'source_commit':original['source_commit'],'production_math_sources':original['production_math_sources'],'directory_name':name,'stage':'hf500_only_actual_formal_no_future_reload','formal_job':'2999','checkpoint_experiment':'opsd-gpu-formal500','checkpoint_step':500,'preview_blocked':True,'file_sources':{n:sha(out/n) for n in files},'original_hf_manifest_sha256':sha(OLD/'source-manifest.json'),'formal_manifest_sha256':sha(FORMAL/'source-manifest.json'),'actual_formal_completion_receipt_required':True,'future_reload_job_id':None}
 (out/'source-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 return m
if __name__=='__main__':assert len(sys.argv)==2;render(sys.argv[1])
