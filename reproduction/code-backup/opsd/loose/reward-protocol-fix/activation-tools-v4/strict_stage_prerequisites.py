"""Local BLOCKED preview: reuse actual training gates; never accept future receipts.

This is not the deployable verifier expected by corrected_evaluation.load_stage.
Its executable entry always exits78. Actual500/HF500/runtime supplements do not exist.
"""
import argparse,hashlib,json,math,re,struct,sys
from pathlib import Path
from training_evidence import loads,digest,own_file,verify_slurm,COMMIT,OWN
import formal_completion_check as formal
from pilot_reload_prerequisites import verify_reload

EXPECTED_DEPENDENCY_BUNDLE_SHA='54457d326b42fba199ea60c7251f81451849e52d41d17b9859dc1447b894a2b3'
def verify_dependency_bundle():
 import sys
 directory=Path(__file__).resolve().parent;manifest=directory/'verifier-dependencies.json'
 if digest(manifest)!=EXPECTED_DEPENDENCY_BUNDLE_SHA:raise ValueError('Strict verifier dependency manifest differs from pinned source')
 records=loads(manifest.read_text())
 for name,sha in records.items():
  path=(directory/name).resolve()
  if not path.is_relative_to(directory) or digest(path)!=sha:raise ValueError('Strict verifier helper dependency bytes changed')
  if name.endswith('.py') and '/' not in name:
   module=sys.modules.get(name[:-3])
   if module is not None and Path(module.__file__).resolve()!=path:raise ValueError('Strict verifier actual imported helper outside reviewed bundle')
 return records

PREVIEW_BLOCKED=True
MODEL_ROLES={'base':OWN/'assets-root/assets/qwen2.5-3b','hf500':OWN/'runs/opsd-formal-3007/output/checkpoints/opsd-gpu-formal500/actor/global_step_500'}

def bound_file(record,scope):
 if not isinstance(record,dict) or set(record)!={'path','sha256'}:raise ValueError('Missing exact path/SHA evidence binding')
 if not re.fullmatch('[0-9a-f]{64}',record['sha256']):raise ValueError('Invalid SHA binding')
 p=own_file(record['path'],scope)
 if digest(p)!=record['sha256']:raise ValueError('Evidence differs from external SHA')
 return p

def read_receipt(path,expected,role,model,base=OWN):
 if role not in MODEL_ROLES:raise ValueError('Unknown model role')
 wanted=base/MODEL_ROLES[role].relative_to(OWN)
 if Path(model)!=wanted:raise ValueError('Model path does not match fixed base/HF500 role')
 p=bound_file(dict(path=str(path),sha256=expected),base)
 r=loads(p.read_text())
 fields={'schema','training_commit','formal_job','eval_commit','formal_completion_receipt','hf500_reload_job','hf500_evidence','model_roles','runtime_supplement'}
 if set(r)!=fields or r['schema']!='corrected-evaluation-layered-prerequisites-v1':raise ValueError('Wrong layered receipt schema')
 if r['training_commit']!=COMMIT or r['formal_job']!='3007':raise ValueError('Training source/job mismatch')
 if not re.fullmatch('[0-9a-f]{40}',r['eval_commit']) or r['eval_commit']==COMMIT:raise ValueError('Corrected evaluation source is missing or substituted by training source')
 if set(r['model_roles'])!={'base','hf500'}:raise ValueError('Both fixed model roles required')
 for name,record in r['model_roles'].items():
  if set(record)!={'directory','files'} or Path(record['directory'])!=base/MODEL_ROLES[name].relative_to(OWN):raise ValueError('Model role directory mismatch')
  verify_model_files(record)
 return r,p

def verify_model_files(record):
 directory=Path(record['directory']);pins=record['files']
 if not isinstance(pins,dict) or not pins:raise ValueError('Missing actual complete model file pins')
 files={p.name for p in directory.iterdir() if p.is_file()}
 if set(pins)!=files or not {'config.json','tokenizer.json'}<=files:raise ValueError('Model content set incomplete')
 for name,sha in pins.items():
  if Path(name).name!=name:raise ValueError('Model file escapes role')
  bound_file(dict(path=str(directory/name),sha256=sha),directory)

def verify_upstream_sources():
 directory=Path(__file__).parent/'upstream';manifest=loads((directory/'source-manifest.json').read_text())
 for name,sha in manifest['file_sources'].items():
  if digest(directory/name)!=sha:raise ValueError('Frozen upstream helper bytes changed')

def verify_training(r,base=OWN):
 # Original immutable verifier checks retained source8e, actual1..500×8 loss/budgets,
 # initial/per50 each64, oldfinal51713 structure, normal terminal and bound review.
 receipt=bound_file(r['formal_completion_receipt'],base)
 proof=formal.verify(receipt,base)
 retained=loads(receipt.read_text());terminal=bound_file(retained['evidence']['slurm_terminal'],base/'runs/opsd-formal-3007').read_text()
 if re.findall(r'\bJobId=([^\s]+)',terminal)!=['3007']:raise ValueError('Formal terminal is not exact own job3007')
 return proof

def shard_keys(checkpoint):
 """CPU header inspection; no torch/GPU load. Actual finite/loading remains raw HF proof."""
 declared=loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map']
 if len(declared)!=435 or len(set(declared.values()))!=3:raise ValueError('Expected435 keys in three actual HF shards')
 actual={}
 for name in set(declared.values()):
  if Path(name).name!=name:raise ValueError('Shard path escapes checkpoint')
  p=own_file(str(checkpoint/name),checkpoint)
  with p.open('rb') as f:
   length=struct.unpack('<Q',f.read(8))[0]
   if not 0<length<=16*1024*1024:raise ValueError('Invalid safetensors header length')
   header=loads(f.read(length))
  for key in header:
   if key=='__metadata__':continue
   if key in actual:raise ValueError('Duplicate actual tensor key')
   actual[key]=name
 if actual!=declared:raise ValueError('Actual shard keys differ from435-key index')
 return actual

def verify_hf500(r,base=OWN):
 job=r['hf500_reload_job']
 if not isinstance(job,str) or not re.fullmatch('[0-9]+',job) or job in {'2994','2996','2999','3007'}:raise ValueError('Real new HF500 reload job missing or old job substituted')
 run=base/('runs/opsd-formal-hf-reload-'+job);checkpoint=base/MODEL_ROLES['hf500'].relative_to(OWN)
 roles={'raw_log','result','source_commit','slurm_terminal','exit','cleanup','checker','checker_manifest','job_script','resources','activation_result','activation_raw','independent_review'}
 if not isinstance(r['hf500_evidence'],dict) or set(r['hf500_evidence'])!=roles:raise ValueError('Missing complete HF500 evidence roles')
 paths={name:bound_file(record,base if name in {'checker','checker_manifest','job_script','independent_review'} else run) for name,record in r['hf500_evidence'].items()}
 if paths['source_commit'].read_text().strip()!=COMMIT or paths['exit'].read_text().strip()!='0':raise ValueError('HF source/exit mismatch')
 terminal=paths['slurm_terminal'].read_text();verify_slurm(terminal,job)
 if re.findall(r'\bJobId=([^\s]+)',terminal)!=[job]:raise ValueError('HF terminal job mismatch')
 for key,value in (('NodeList','h100-10'),('NumNodes','1')):
  if re.findall(r'\b'+key+r'=([^\s]+)',terminal)!=[value]:raise ValueError('HF same-node allocation mismatch')
 if re.findall(r'\bTresPerNode=([^\s]+)',terminal)!=['gres:gpu:h100:1']:raise ValueError('HF own1GPU lease missing')
 template_manifest=loads((Path(__file__).parent/'source-manifest.json').read_text())
 from pre_stage_runtime import verify_hf_activation
 verify_hf_activation(paths['checker_manifest'],paths['job_script'],paths['checker'],r['hf500_evidence']['checker_manifest']['sha256'])
 from activation_verification import verify_activation_evidence
 verify_activation_evidence(paths,job,base)
 # Job script activation removes guard, so future script must be reviewed separately.
 # Do not pretend the blocked original script was executed unchanged.
 result=loads(paths['result'].read_text());verify_reload(result,checkpoint,job)
 if result['tensor_count']!=435 or len(result['weights'])!=3:raise ValueError('Incomplete435-key/3-shard HF reload')
 shard_keys(checkpoint)
 marker='REAL_3B_HF_CHECKPOINT_RELOAD_AND_STUDENT_CHANGE_OK '
 lines=[line.split(marker,1)[1] for line in paths['raw_log'].read_text().splitlines() if marker in line]
 if len(lines)!=1 or loads(lines[0])!=result:raise ValueError('HF raw checker output/result mismatch')
 cleanup=loads(paths['cleanup'].read_text())
 if cleanup.get('job_id')!=job or cleanup.get('node_scope')!='h100-10' or cleanup.get('previous_training_job')!='3007' or cleanup.get('previous_job_processes')!=[] or cleanup.get('previous_ports_listening') is not False:raise ValueError('Prior training cleanup missing')
 review=loads(paths['independent_review'].read_text())
 expected={key:record['sha256'] for key,record in r['hf500_evidence'].items() if key!='independent_review'}
 if review.get('source_commit')!=COMMIT or review.get('formal_job')!='3007' or review.get('reload_job')!=job or review.get('bound_evidence_sha256')!=expected or review.get('accepted') is not True or not review.get('reviewer') or not review.get('report'):raise ValueError('Independent HF500 review not bound to all actual evidence')
 return result

def verify_runtime_supplement(r,base=OWN,actual_stage_path=None,actual_stage_sha=None,model_role=None,receipt_path=None,receipt_sha=None):
 from pre_stage_runtime import verify_pre_stage
 supplement=bound_file(r['runtime_supplement'],base)
 verify_pre_stage(loads(supplement.read_text()),r,base,actual_stage_path,actual_stage_sha,model_role,receipt_path,receipt_sha)
 return True

def verify(path,expected,role,model,base=OWN,actual_stage_path=None,actual_stage_sha=None):
 verify_dependency_bundle();verify_upstream_sources()
 r,p=read_receipt(path,expected,role,model,base)
 verify_training(r,base);verify_hf500(r,base);verify_runtime_supplement(r,base,actual_stage_path,actual_stage_sha,role,p,expected)
 # Called only after every actual strict gate; no caller-provided passing flags.
 return dict(kind='strict_actual_training3007_hf500_prerequisites',training_commit=COMMIT,formal_job='3007',actual_outer_updates=500,final_questions=51713,receipt_sha256=digest(p),model_role=role,model_directory=str(model))

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--receipt',required=True);parser.add_argument('--receipt-sha256',required=True);parser.add_argument('--model-role',choices=('base','hf500'),required=True);parser.add_argument('--model-directory',required=True)
 parser.add_argument('--stage',default='UNFILLED');parser.add_argument('--stage-sha256',default='UNFILLED')
 args=parser.parse_args()
 print('BLOCKED PREVIEW: actual500/HF500/runtime supplements UNFILLED; not deployable',file=sys.stderr)
 return 78
 proof=verify(args.receipt,args.receipt_sha256,args.model_role,args.model_directory,actual_stage_path=args.stage,actual_stage_sha=args.stage_sha256)
 print(json.dumps(proof,sort_keys=True,allow_nan=False))
 return 0
if __name__=='__main__':raise SystemExit(main())
