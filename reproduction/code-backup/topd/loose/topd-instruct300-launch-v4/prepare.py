"""Create a future owned student300 config from current local infrastructure."""
import argparse,hashlib,json,pathlib,sys
sys.dont_write_bytecode=True
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def derive(infrastructure,teacher_result,teacher_result_sha256,run_directory):
 c=dict(infrastructure);recipe=json.loads(pathlib.Path(__file__).with_name('recipe.json').read_text())
 source=ROOT/'tools/topd-instruct300-source-v1';tools=ROOT/'tools/topd-instruct300-launch-v4/runtime';backend=ROOT/'tools/topd-vllm-teacher-v2'
 run=pathlib.Path(run_directory)
 if ROOT/'logs' not in run.parents:raise ValueError('Owned fresh run path required')
 c.update(mode='topd300',expected_outer_steps=300,candidate=str(source),tools=str(tools),teacher_backend=str(backend),base_model=str(ROOT/'assets/qwen2.5-3b-instruct-aa8e725'),evidence_root=str(run),checkpoint_root=str(run/'checkpoints'),actual_outer_receipt_path=str(run/'actual-outer-updates.json'),train_file=str(ROOT/'assets/common-instruct-0b002b/train.parquet'),val_file=str(ROOT/'assets/common-instruct-0b002b/validation.parquet'),final_checkpoint=str(run/'checkpoints/actor/global_step_300'),teacher_gpu_memory_utilization=.25)
 # Actual teacher is absent until GRPO100 delivers; no fabricated selection/proof.
 if teacher_result is None:
  c['teacher_actual_result']=None;c['teacher_path']=None
 else:
  if sha(teacher_result)!=teacher_result_sha256:raise ValueError('Actual teacher result SHA differs')
  result=json.loads(pathlib.Path(teacher_result).read_text());c['teacher_actual_result']={'path':str(teacher_result),'sha256':teacher_result_sha256};c['teacher_path']=result['export']['directory']
 c['launcher_manifest']={'path':str(ROOT/'tools/topd-instruct300-launch-v4/files.json'),'sha256':None}
 c['source_manifest']={'path':str(source/'files.json'),'sha256':None};c['tools_manifest']={'path':str(tools/'files.json'),'sha256':None};c['teacher_backend_manifest']={'path':str(backend/'files.json'),'sha256':None}
 c['training_overrides']=recipe['overrides']+['baselines.distillation.teacher_path='+str(c['teacher_path']),'baselines.topd.socket_manifest='+str(run/'teacher/client-config.json'),'trainer.default_local_dir='+str(run/'checkpoints'),'+trainer.topd_actual_outer_receipt='+str(run/'actual-outer-updates.json')]
 common=ROOT/'assets/common-instruct-0b002b'
 identities={'manifest.json':'0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24','train.parquet':'5d40180ba54d35513bb780408caa068d7d60123214a49b0739f835a2ae47c8a1','validation.parquet':'3d6318365b5bb98039f1e28ca6bc7d1de98eccc5cc833600e39ce1c2306f9ceb','stage.parquet':'b529fe2a4b6424091aa4ca827a28337c978808ecde0e01e6fe7c867e14a9084e','final.parquet':'55d515e116a4c6aa63b45282f9bc87202b030156be41be60420c38a9d3ff0ed1'}
 c['input_bindings']=[{'path':str(common/name),'sha256':value} for name,value in identities.items()]

 c['telemetry_policy']={'directory':str(ROOT/'tools/qwen25-32b-window-safety-direct-own-sm-v1'),'runtime_sha256':'9b7ae78787645270c4768ac303b4b206e4c0e79231aa66e077c32847211a81b8','guard_sha256':'5e3c1aa1bb073c308f3bcda2bd31f928708aafea23ee3e33a4ebeb558905ced5','manifest_sha256':'37168c1a220b80f3c7895e9297607a6e988a4f1fadee8db991915e24703dc709'}
 c['student_model_identity']=None
 c['independent_gate']=None
 return c
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--infrastructure',required=True);p.add_argument('--teacher-result');p.add_argument('--teacher-result-sha256');p.add_argument('--run-directory',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 c=derive(json.loads(pathlib.Path(a.infrastructure).read_text()),a.teacher_result,a.teacher_result_sha256,a.run_directory)
 with pathlib.Path(a.output).open('x') as f:json.dump(c,f,indent=2)
