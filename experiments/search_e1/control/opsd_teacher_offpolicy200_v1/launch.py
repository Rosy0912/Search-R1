"""Fresh fixed ordinary source100 teacher rollouts; no legacy Adam reuse."""
import os,sys,json,copy,time,uuid,hashlib,shlex,subprocess,argparse
from pathlib import Path
OLD=Path('/data/shaoshitong/search-e1-opsd-j-minus-p-bf16-continuation200-20261010')
BASE=Path('/data/shaoshitong/search-e1-opsd-fixed-teacher-bf16-rwm200-20261010')
def main():
 p=argparse.ArgumentParser();p.add_argument('--publication',required=True);a=p.parse_args()
 import prepare,gates
 from checkpoint import exclusive_json,fileproof,checked_json,require
 require(os.getuid()==1005 and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Owned CPU launcher required')
 require(not (BASE/'receipts/training-config.json').exists(),'No duplicate fresh launch')
 (BASE/'receipts').mkdir(exist_ok=True);(BASE/'logs').mkdir(exist_ok=True);receipts=BASE/'receipts'
 old=json.loads((OLD/'receipts/training-config.json').read_text());cfg=copy.deepcopy(old);cfg.pop('continuation',None)
 pub=json.loads(Path(a.publication).read_text());cfg['run_id']=os.environ['E1_RUN_ID'];cfg['study_id']='fixed-teacher-offpolicy200-'+cfg['run_id'];cfg['source_commit']=pub['source_commit']
 cfg['paths'].update(base=str(BASE),output_root=str(BASE/'runs'),receipts=str(receipts),helper_root=str(Path(__file__).parent.resolve()))
 cfg['logging']['full_vocab_study_root']=str(Path(cfg['paths']['study_base'])/'runs/full-vocabulary')
 cfg['helper_source_manifest']=fileproof(Path(__file__).parent/'source-manifest.json')
 cfg['trajectory_source']=dict(role='fixed_source100_teacher',prompt='ordinary',fresh_per_outer=True,student_updates_affect_sampling=False)
 cfg['autonomous_diagnostics']=dict(cases_per_outer=1,roles=['student_ordinary','teacher_privileged'],seed=42,temperature=0,training_denominator_contribution=0)
 now=time.time();cfg['authorization']['reverse_seven_group_ten_outer']=dict(human_observed_unix=now,text='2026-10-10: fixed teacher ordinary prompt trajectories, BF16 R+W+M, fresh source100 Adam,200 outer, every10 native512, follow first5 and real6')
 scope=dict(study_id=cfg['study_id'],variant='J',run_id=cfg['run_id'],output_root=cfg['paths']['output_root'])
 cfg['gates']['publication']=exclusive_json(receipts/'publication.json',dict(pub,scope=scope,source_manifest_sha256=cfg['helper_source_manifest']['sha256']))
 for name in ('storage','retriever_admission'):
  value=copy.deepcopy(checked_json(old['gates'][name])[0]);value['scope']=scope;value['observed_unix']=now
  if name=='storage':value['output_root']=cfg['paths']['output_root'];value['backup_policy']['durable_target']=str(BASE/'runs/checkpoints')
  cfg['gates'][name]=exclusive_json(receipts/(name+'.json'),value)
 devices,apps=gates.inventory();require(all(x['pid']==2239529 for x in apps),'Previous GPU workers remain or foreign process')
 cfg['gates']['resource_transition']=exclusive_json(receipts/'resource-transition.json',dict(accepted=True,cluster='HKUST',scope=scope,gpu_uuids=cfg['owner']['gpu_uuids'],actual_devices=devices,actual_applications=apps,observed_unix=now))
 cfg['gates']['memory_binding_path']=str(receipts/'memory-binding.json');cfg['gates']['scoped_cpu_processes_path']=str(receipts/'scoped-cpu-processes.json')
 prepare.validate_config(cfg,launch=True);ref=exclusive_json(receipts/'training-config.json',cfg);cfg['_config_sha256']=ref['sha256']
 registry=receipts/'scoped-cpu-processes.json';registry.write_text(json.dumps(dict(schema='opsd-ablation-scoped-cpu-processes-v1',accepted=True,run_id=cfg['run_id'],config_sha256=ref['sha256'],roles=[])))
 gates.require_launch_without_memory(cfg,cfg['run_id'],gates.process(os.getpid()))
 source=Path(__file__).parent/'memory_observer.py';marker='fixed-teacher-memory-'+str(int(now))
 cmd='env E1_RUN_ID='+shlex.quote(marker)+" CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "+shlex.quote(cfg['paths']['python'])+' -B '+shlex.quote(str(source))+' --config '+shlex.quote(ref['path'])+' --output '+str(receipts/'memory-live.json')+' --binding-output '+str(receipts/'memory-binding.json')+' >> '+str(BASE/'logs/memory.log')+' 2>&1'
 subprocess.run(['tmux','new-session','-d','-s','e1-fixed-teacher-memory-20261010',cmd],check=True)
 for _ in range(30):
  if (receipts/'memory-binding.json').exists() and (receipts/'memory-live.json').exists():break
  time.sleep(1)
 observer=json.loads((receipts/'memory-binding.json').read_text())['observer_identity'];identity=gates.process(observer['pid'])
 role=dict(role='memory_observer',identity=identity,no_gpu_or_training=True,argv_sha256=hashlib.sha256(json.dumps(identity['argv'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),input_config=ref,helper_source_manifest=cfg['helper_source_manifest'],publication=cfg['gates']['publication'],source=fileproof(source))
 registry.write_text(json.dumps(dict(schema='opsd-ablation-scoped-cpu-processes-v1',accepted=True,run_id=cfg['run_id'],config_sha256=ref['sha256'],roles=[role])))
 exclusive_json(receipts/'launch.json',dict(accepted=True,run_id=cfg['run_id'],config=ref,controller_pid=os.getpid(),full_budget=200,initial_acceptance_only=5,evaluation_every=10,trajectory_source=cfg['trajectory_source']))
 os.execv(cfg['paths']['python'],[cfg['paths']['python'],'-B',str(Path(__file__).parent/'controller.py'),'--config',ref['path'],'--config-sha256',ref['sha256'],'--run-id',cfg['run_id']])
if __name__=='__main__':main()
