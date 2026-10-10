"""Refresh source bindings for the idle four-GPU draft; no submission."""
import ast,hashlib,json,pathlib,subprocess,sys
BASE=pathlib.Path('/data/user/jxu957/experiments/topd-20261008');ROOT=BASE/'runtime'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 if subprocess.check_output(['squeue','-u','mou176','-h','-o','%j'],universal_newlines=True).lower().find('topd')>=0:raise RuntimeError('Active job prevents mutation')
 rt=ROOT/'tools/topd-instruct300-launch-v6/runtime'
 for p in [rt/'supervise.py',rt/'owner_support.py',ROOT/'tools/search-r1-32b-grpo100-tools-direct-v5/owner_support.py']:
  s=p.read_text().replace("<96","<48").replace('>=96','>=48').replace('<768*1024','<384*1024')
  ast.parse(s);p.write_text(s)
 stage=rt/'student_phase.py'
 s=stage.read_text().replace('stage.parquet','validation.parquet').replace('stage64','stage512').replace('STAGE64','STAGE512').replace('!=64','!=512').replace('batch_size=64','batch_size=512').replace("'rows':64","'rows':512")
 ast.parse(s);stage.write_text(s)
 launch=rt.parent/'launch.sbatch'
 s=launch.read_text().replace('--job-name=topd-instruct300','--job-name=topd-instruct200').replace('--gres=gpu:8','--gres=gpu:4').replace('--cpus-per-task=96','--cpus-per-task=48').replace('--mem=768G','--mem=384G')
 launch.write_text(s)
 c=json.loads((BASE/'outputs/training-preparation/hpc3-topd200-protocol-draft.json').read_text())
 for key in ['source_manifest','teacher_backend_manifest','tools_manifest','launcher_manifest']:
  p=pathlib.Path(c[key]['path']);root=p.parent;old=json.loads(p.read_text())
  old['files']={str(f.relative_to(root)):{'size':f.stat().st_size,'sha256':sha(f)} for f in root.rglob('*') if f.is_file() and f!=p and '__pycache__' not in f.parts}
  p.write_text(json.dumps(old,indent=2,sort_keys=True));c[key]['sha256']=sha(p)
 for key in ['support','preflight_source']:c[key]['sha256']=sha(c[key]['path'])
 c['preparation_status']='FOUR_GPU_RUNTIME_BINDINGS_READY_FOR_STATIC_VALIDATION_NOT_GPU_ACCEPTED'
 out=BASE/'outputs/training-preparation/hpc3-topd200-protocol-draft.json';out.write_text(json.dumps(c,indent=2))
 sys.path.insert(0,str(rt))
 from supervise import verify_recipe
 from student_binding import check
 assert verify_recipe(c)==200
 check(c)
 print('FOUR_GPU_PROTOCOL_AND_EXACT_SOURCE_BINDINGS_PASS')
if __name__=='__main__':main()
