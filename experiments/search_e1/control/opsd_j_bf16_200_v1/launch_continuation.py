"""Extend accepted five-round checkpoint to total200, without evaluation or optimizer reset."""
import os,sys,json,copy,time,hashlib,shlex,subprocess,argparse
from pathlib import Path
OLD=Path('/data/shaoshitong/search-e1-opsd-j-minus-p-bf16-20261010');BASE=Path('/data/shaoshitong/search-e1-opsd-j-minus-p-bf16-continuation200-20261010')
def main():
 p=argparse.ArgumentParser();p.add_argument('--publication',required=True);a=p.parse_args();import prepare,gates
 from checkpoint import fileproof,exclusive_json,checked_json,check_checkpoint,require
 from cohort_v1 import freeze_from_files
 require(os.getuid()==1005 and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only owned launcher required')
 oldraw=(OLD/'receipts/training-config.json').read_bytes();old=json.loads(oldraw);accept=json.loads((OLD/'receipts/five-outers-accepted-cpu.json').read_text());require(accept['accepted'] and accept['outers']==5 and accept['real_Adam']==50 and accept['run_id']==old['run_id'],'Actual CP50 acceptance required')
 status=json.loads((OLD/'runs/status.json').read_text());require(status['phase']=='TRAIN_COMPLETE' and not Path('/proc/'+str(status['owner']['pid'])).exists(),'Old controller must truly be finished')
 require(not BASE.exists(),'Never overwrite continuation preparation');(BASE/'receipts').mkdir(parents=True);(BASE/'logs').mkdir();receipts=BASE/'receipts'
 pub=json.loads(Path(a.publication).read_text());require(pub['accepted'] and pub['source_commit']==pub['remote_readback_sha'],'Published new source required')
 cfg=copy.deepcopy(old);cfg['source_commit']=pub['source_commit'];cfg['scientific']['outer_iterations']=200
 cfg['paths'].update(base=str(BASE),output_root=str(BASE/'runs'),receipts=str(receipts),helper_root=str(Path(__file__).parent.resolve()))
 cfg['sampling']['question_order']='shared_frozen_original_seed42_first102400_train_permutation_no_replacement';cfg['mining']['cohort_version']='search-e1-opsd-j-bf16-200-train-cohorts-seed42-v1'
 frozen=freeze_from_files(cfg['train_identity'],cfg['validation_identity']);original=checked_json(old['common_cohort_manifest'])[0]
 require(frozen['cohorts'][:10]==original['cohorts'] and frozen['permutation_sha256']==original['permutation_sha256'],'Original first ten TRAIN cohorts changed')
 cfg['common_cohort_manifest']=exclusive_json(receipts/'common-cohorts200.json',frozen);cfg['helper_source_manifest']=fileproof(Path(__file__).parent/'source-manifest.json')
 cfg['continuation']=dict(prefix_checkpoint=accept['final_checkpoint'],prefix_config_sha256=hashlib.sha256(oldraw).hexdigest(),prefix_config=fileproof(OLD/'receipts/training-config.json'),prefix_output_root=str(OLD/'runs'),prefix_acceptance=fileproof(OLD/'receipts/five-outers-accepted-cpu.json'),completed_outers=5,optimizer_updates=50,cumulative_pair_exposure=1497,next_outer_index=5,full_budget=200,no_evaluation=True,no_optimizer_reset=True,no_prefix_regeneration=True)
 now=time.time();cfg['authorization']['reverse_seven_group_ten_outer']=dict(human_observed_unix=now,text='2026-10-10 human correction: five outers are initial acceptance only; continue same weights/full Adam/RNG/cursor to total200, no512 evaluation, no repeated five-round tests.')
 scope=dict(study_id=cfg['study_id'],variant='J',run_id=cfg['run_id'],output_root=cfg['paths']['output_root'])
 cfg['gates']['publication']=exclusive_json(receipts/'publication.json',dict(pub,scope=scope,source_manifest_sha256=cfg['helper_source_manifest']['sha256']))
 for name,key in [('storage','storage'),('retriever_admission','oracle18004-admission')]:
  value=checked_json(old['gates'][name])[0];value=copy.deepcopy(value);value['scope']=scope;value['observed_unix']=now
  if name=='storage':value['output_root']=cfg['paths']['output_root'];value['backup_policy']['durable_target']=str(BASE/'runs/checkpoints')
  cfg['gates'][name]=exclusive_json(receipts/(key+'.json'),value)
 gates.retriever_admission(cfg,checked_json(cfg['gates']['retriever_admission'])[0]);devices,apps=gates.inventory();require([d['uuid'] for d in devices]==cfg['owner']['gpu_uuids'] and all(x['pid']==2239529 for x in apps),'Owned GPUs not released')
 cfg['gates']['resource_transition']=exclusive_json(receipts/'resource-transition.json',dict(accepted=True,cluster='HKUST',scope=scope,gpu_uuids=cfg['owner']['gpu_uuids'],actual_devices=devices,actual_applications=apps,observed_unix=now,prefix=cfg['continuation']))
 cfg['gates']['memory_binding_path']=str(receipts/'memory-binding.json');cfg['gates']['scoped_cpu_processes_path']=str(receipts/'scoped-cpu-processes.json')
 prepare.validate_config(cfg,launch=True);ref=exclusive_json(receipts/'training-config.json',cfg);cfg['_config_sha256']=ref['sha256'];check_checkpoint(cfg['continuation']['prefix_checkpoint'],cfg,cfg['run_id'],50,4)
 registry=receipts/'scoped-cpu-processes.json';registry.write_text(json.dumps(dict(schema='opsd-ablation-scoped-cpu-processes-v1',accepted=True,run_id=cfg['run_id'],config_sha256=ref['sha256'],roles=[])))
 gates.require_launch_without_memory(cfg,cfg['run_id'],gates.process(os.getpid()))
 source=Path(__file__).parent/'memory_observer.py';marker='j-bf16-200-memory-'+str(int(now));command='env E1_RUN_ID='+shlex.quote(marker)+" CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "+shlex.quote(cfg['paths']['python'])+' -B '+shlex.quote(str(source))+' --config '+shlex.quote(ref['path'])+' --output '+str(receipts/'memory-live.json')+' --binding-output '+str(receipts/'memory-binding.json')+' >> '+str(BASE/'logs/memory-observer.log')+' 2>&1'
 subprocess.run(['tmux','new-session','-d','-s','e1-j-bf16-200-memory-20261010',command],check=True)
 for _ in range(30):
  if (receipts/'memory-binding.json').exists() and (receipts/'memory-live.json').exists():break
  time.sleep(1)
 observer=json.loads((receipts/'memory-binding.json').read_text())['observer_identity'];identity=gates.process(observer['pid']);require(identity['ppid']!=os.getpid(),'Observer must be independent')
 role=dict(role='memory_observer',identity=identity,no_gpu_or_training=True,argv_sha256=hashlib.sha256(json.dumps(identity['argv'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),input_config=ref,helper_source_manifest=cfg['helper_source_manifest'],publication=cfg['gates']['publication'],source=fileproof(source))
 registry.write_text(json.dumps(dict(schema='opsd-ablation-scoped-cpu-processes-v1',accepted=True,run_id=cfg['run_id'],config_sha256=ref['sha256'],roles=[role])))
 exclusive_json(receipts/'launch.json',dict(accepted=True,run_id=cfg['run_id'],config=ref,controller_pid=os.getpid(),full_budget=200,initial_acceptance_only=5,resume_from=cfg['continuation'],no512_evaluation=True))
 print(json.dumps(dict(config=ref,resume_from=cfg['continuation'],controller_pid=os.getpid())),flush=True)
 os.execv(cfg['paths']['python'],[cfg['paths']['python'],'-B',str(Path(__file__).parent/'controller.py'),'--config',ref['path'],'--config-sha256',ref['sha256'],'--run-id',cfg['run_id']])
if __name__=='__main__':main()
