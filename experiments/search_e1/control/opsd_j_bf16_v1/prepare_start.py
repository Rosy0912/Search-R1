"""Standalone human-authorized J-minus-P preparation, observer, then five-outer controller."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time,uuid
from pathlib import Path

def proof(p):
 p=Path(p).resolve();raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,sort_keys=True,indent=2);f.write('\n')
 return proof(p)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--publication',required=True);a=parser.parse_args()
 import prepare,gates
 assert os.getuid()==1005 and os.environ.get('CUDA_VISIBLE_DEVICES')==''
 base=Path('/data/shaoshitong/search-e1-opsd-j-minus-p-bf16-20261010');receipts=base/'receipts';receipts.mkdir(parents=True,exist_ok=True);(base/'logs').mkdir(exist_ok=True)
 stop=receipts/'old-pr-stop-delete.json';old_stop=json.loads(stop.read_text());assert not old_stop['survivors'] and old_stop['all_deleted_absent']
 old=json.loads(Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/group-R/receipts/authorized-launch-20261009/config.json').read_text())
 pub=json.loads(Path(a.publication).read_text());assert pub['accepted'] and pub['source_commit']==pub['remote_readback_sha']
 cfg=prepare.template(str(base),'J','j-minus-p-bf16-'+uuid.uuid4().hex,str(base));cfg['run_id']=uuid.uuid4().hex;cfg['source_commit']=pub['source_commit'];cfg['paths']['helper_root']=str(Path(__file__).resolve().parent);cfg['paths']['runtime']=old['paths']['runtime']
 for k in ['source_model_manifest','runtime_manifest','train_identity','validation_identity','common_cohort_manifest','common_initial_adapter_manifest','common_initial_adapter_tensor_sha256']:cfg[k]=old[k]
 cfg['helper_source_manifest']=proof(Path(__file__).parent/'source-manifest.json')
 now=time.time();cfg['authorization']['reverse_seven_group_ten_outer']={'human_observed_unix':now,'text':'2026-10-10 latest human: stop/delete current PR; new J-minus-P BF16 R+W+M; five actual online outers before acceptance. Legacy authorization field name only, not seven-group or ten-outer execution.'};cfg['authorization']['eight_gpu_execution']={'node':'hkust1','cluster':'HKUST','uid':1005,'human_observed_unix':now}
 scope=dict(study_id=cfg['study_id'],variant='J',run_id=cfg['run_id'],output_root=cfg['paths']['output_root'])
 cfg['gates']['publication']=save(receipts/'publication.json',dict(pub,scope=scope,source_manifest_sha256=cfg['helper_source_manifest']['sha256']))
 cfg['gates']['storage']=save(receipts/'storage.json',dict(accepted=True,cluster='HKUST',mode='human_hkust_local_exception',scope=scope,output_root=cfg['paths']['output_root'],human_authorization=dict(explicit_exception=True,observed_unix=now,thread_id='01a1028b-b626-7f91-93db-0099ff1623c5',text='HKUST node local storage explicitly authorized; keep model/state remotely, only code/config/small reports on Mac'),backup_policy=dict(durable_target=str(base/'runs/checkpoints'),synchronize_completed_checkpoints=True,Mac_payloads=False,same_node_not_disaster_recovery=True)))
 oracle=json.loads(Path(old['gates']['retriever_admission']['path']).read_text());oracle.update(scope=scope,observed_unix=now);gates.retriever_admission(cfg,oracle)
 devices,apps=gates.inventory();assert [d['uuid'] for d in devices]==cfg['owner']['gpu_uuids'];assert all(x['pid']==oracle['pid'] for x in apps),apps
 cfg['gates']['resource_transition']=save(receipts/'resource-transition.json',dict(accepted=True,cluster='HKUST',scope=scope,gpu_uuids=cfg['owner']['gpu_uuids'],actual_devices=devices,actual_applications=apps,old_stop_delete=proof(stop),observed_unix=now))
 cfg['gates']['retriever_admission']=save(receipts/'oracle18004-admission.json',oracle)
 for name in cfg['resources']['shared_gpu_lease_paths']+[cfg['resources']['study_lease_path']]:Path(name).parent.mkdir(parents=True,exist_ok=True)
 prepare.validate_config(cfg,launch=True);ref=save(receipts/'training-config.json',cfg);gate_cfg=dict(cfg,_config_sha256=ref['sha256']);os.environ['E1_RUN_ID']=cfg['run_id']
 registry=Path(cfg['gates']['scoped_cpu_processes_path']);registry.write_text(json.dumps(dict(schema='opsd-ablation-scoped-cpu-processes-v1',accepted=True,run_id=cfg['run_id'],config_sha256=ref['sha256'],roles=[])))
 gates.require_launch_without_memory(gate_cfg,cfg['run_id'],gates.process(os.getpid()))
 observer_source=Path(cfg['paths']['helper_root'])/'memory_observer.py';env=dict(os.environ,E1_RUN_ID='j-bf16-memory-'+uuid.uuid4().hex,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
 log=(base/'logs/memory-observer.log').open('x');child=subprocess.Popen([cfg['paths']['python'],'-B',str(observer_source),'--config',ref['path'],'--output',str(receipts/'memory-live.json'),'--binding-output',str(receipts/'memory-binding.json')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 for _ in range(30):
  if (receipts/'memory-binding.json').exists() and (receipts/'memory-live.json').exists():break
  if child.poll() is not None:raise RuntimeError('Observer exited '+str(child.returncode))
  time.sleep(1)
 identity=gates.process(child.pid);role=dict(role='memory_observer',identity=identity,no_gpu_or_training=True,argv_sha256=hashlib.sha256(json.dumps(identity['argv'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),input_config=ref,helper_source_manifest=cfg['helper_source_manifest'],publication=cfg['gates']['publication'],source=proof(observer_source))
 registry.write_text(json.dumps(dict(schema='opsd-ablation-scoped-cpu-processes-v1',accepted=True,run_id=cfg['run_id'],config_sha256=ref['sha256'],roles=[role])))
 gates.require_launch(gate_cfg,'cpu-merge',cfg['run_id'],gates.process(os.getpid()),[])
 save(receipts/'launch.json',dict(accepted=True,training_started=False,config=ref,run_id=cfg['run_id'],controller_pid=os.getpid(),observer_pid=child.pid,expected_real_outers=5,precision='bf16',mechanisms=['R','W','M'],source_commit=cfg['source_commit']))
 print(json.dumps({'config':ref,'run_id':cfg['run_id'],'BF16_only':True,'launch_is_not_acceptance':True}),flush=True)
 os.execv(cfg['paths']['python'],[cfg['paths']['python'],'-B',str(Path(cfg['paths']['helper_root'])/'controller.py'),'--config',ref['path'],'--config-sha256',ref['sha256'],'--run-id',cfg['run_id']])
if __name__=='__main__':main()
