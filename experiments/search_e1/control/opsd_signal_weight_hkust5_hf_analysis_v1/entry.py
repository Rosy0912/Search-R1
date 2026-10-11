"""Private K5 resource/input overlay; original probability and statistical math unchanged."""
import argparse,copy,fcntl,hashlib,importlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
S=Path('/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010')
HF_ROOT=S/'source/hkust5-hf-eval-0b68ff1/opsd_signal_weight_hkust5_hf_eval_v1'
HF_PUB={'path':str(S/'receipts/hkust5-hf-evaluator-publication-0b68ff1.json'),'sha256':'cac35add8f07675e8b9fda211ab95ee717ef1bfaeffa566b26f6e3eced260c21'}
BINDING={'path':str(S/'receipts/hkust5-isolation-binding-v2.json'),'sha256':'cdd237b50439d7caed5388f3d5cacad43d8fe0f978aa6b11c7c6c40e81bd2713'}
G_TERMINALS={
 'G_POS':{'path':str(S/'sequence-attempts/G_POS-d92dbb708aef4f5c86e866e3fe572cf8/actual-controller-exit.json'),'sha256':'a2ef9a798a81a274345715d707b30a9e45d47c075b7381db7fa557d276f2d7b1'},
 'G_NEG':{'path':str(S/'sequence-attempts/G_NEG-a6725c9c78d54ca68f2b1972ed82206e/actual-controller-exit.json'),'sha256':'91a148d4eb20d7b25644db8b213024b8eb984762530c8290006cb2660fc02dbd'}}
TARGETS=('adapter.py','replay.py','replay_worker.py','summary.py','em_summary.py','build_terminal.py','integration_cpu.py')

def require(ok,message):
 if not ok:raise ValueError(message)
def pinned(ref):
 raw=Path(ref['path']).read_bytes();require(hashlib.sha256(raw).hexdigest()==ref['sha256'],'Pinned identity changed');return json.loads(raw)
def verify_helper(root,pubref):
 root=Path(root);pub=pinned(pubref);raw=(root/'source-manifest.json').read_bytes()
 require(pub['accepted'] and pub['branch']=='search-e1' and pub['source_commit']==pub['remote_readback_sha'] and len(pub['source_commit'])==40 and hashlib.sha256(raw).hexdigest()==pub['source_manifest_sha256'],'Unpublished helper')
 require(Path(pub['deployed_helper']).resolve()==root.resolve(),'Helper publication names another directory')
 manifest=json.loads(raw);names=manifest['files']
 require({p.name for p in root.glob('*.py')}=={n for n in names if n.endswith('.py')},'Python inventory changed')
 for name,item in names.items():
  require(Path(name).name==name and hashlib.sha256((root/name).read_bytes()).hexdigest()==item['sha256'],'Helper source changed')
 return pub

def validate_devices(config,devices,apps,owner,run_id,process,owned_process,released):
 require([d['index'] for d in devices]==[0,1,2,3] and [d['uuid'] for d in devices]==config['owner']['gpu_uuids'],'K5 allocation changed')
 owned=[]
 for app in apps:
  require(app['pid']!=owner['pid'],'CPU parent initialized CUDA')
  # The oracle is remote: even a coincidentally equal K5 PID is NOT exempt.
  actual=owned_process(app['pid'],run_id,owner,[]);owned.append(actual)
 require(not released or not owned,'Prior owned GPU process remains')
 floor=config['resources']['startup_minimum_free_mib'] if released else config['resources']['training_minimum_free_mib']
 require(all(d['actual_free_mib']>=floor for d in devices),'Direct memory.free below original floor')
 return owned

def setup(publication_ref):
 verify_helper(ROOT,publication_ref);verify_helper(HF_ROOT,HF_PUB)
 sys.path.insert(0,str(HF_ROOT))
 spec=importlib.util.spec_from_file_location('_signal_k5_HF_resources',HF_ROOT/'entry.py');hf=importlib.util.module_from_spec(spec);sys.modules[spec.name]=hf;spec.loader.exec_module(hf)
 binding=pinned(BINDING);migration_path=S/'source/hkust5-device-isolation-v2/opsd_signal_weight_hkust5_migration_v1/entry.py'
 spec=importlib.util.spec_from_file_location('_signal_k5_migration_resources',migration_path);migration=importlib.util.module_from_spec(spec);sys.modules[spec.name]=migration;spec.loader.exec_module(migration)
 migration.install(binding);sys.path.insert(0,str(ROOT))
 for name,digest in json.loads((ROOT/'frozen-files.json').read_text())['files'].items():require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,'Frozen analysis body changed')
 import common
 native_read=common.read_json;native_bind=common.bind_training
 def read(ref):
  value,proof=native_read(ref)
  if value.get('schema')=='opsd-signal-weight-config-v1':
   require(proof['sha256']==hf.SHAS[value['variant']],'Immutable arm config differs');value=copy.deepcopy(value);value['owner']=dict(uid=1001,physical_gpus=[0,1,2,3],gpu_uuids=migration.UUIDS[:])
  return value,proof
 common.read_json=read
 def bind(config):
  modules=native_bind(config);modules['gates'].inventory=migration.inventory
  return modules
 common.bind_training=bind
 import replay
 def boundary(config,modules,owner,run_id,released=False):
  gates=modules['gates'];require(gates.same_process(gates.process(owner['pid']),owner) and owner['uid']==1001 and owner['run_id']==run_id,'Replay owner changed')
  oracle=migration.oracle(config,gates);devices,apps=migration.inventory()
  owned=validate_devices(config,devices,apps,owner,run_id,gates.process,gates.owned_gpu_process,released)
  return dict(accepted=True,checked_unix=time.time(),devices=devices,owned=owned,oracle=oracle,released=released,no_signals_sent=True)
 def release(config,modules,owner,run_id,children,helper_root,output):
  release_config=copy.deepcopy(config);release_config['paths']['output_root']=str(output)
  return migration.release(release_config,run_id,owner,children)
 def terminal(value,modules):
  actual=hf.terminal()  # Exact same current migration/server/foreground terminal0 guard.
  require(value.get('schema')=='opsd-signal-k5-analysis-terminal-v1' and value['accepted'] and value['migration_binding']==BINDING,'Analysis lineage differs')
  require(pinned(value['migration_training_complete'])==actual,'Actual all100 completion differs')
  require(value['G_actual_terminal']==G_TERMINALS,'Original G wait refs differ')
  for g in hf.SHAS:
   cfg,_=read(value['configs'][g]);require(value['configs'][g]['sha256']==hf.SHAS[g],'Config lineage differs')
   ledger=pinned(value['train_ledgers'][g]);require(ledger['accepted'] and ledger['outer_iterations']==25 and ledger['variant']==g and len(ledger['rows'])==25,'Incomplete TRAIN ledger')
   require([r['outer_index'] for r in ledger['rows']]==list(range(25)) and all(r['train_denominator']==2560 and r['config_sha256']==hf.SHAS[g] for r in ledger['rows']),'TRAIN population/config differs')
   if g in G_TERMINALS:t=pinned(G_TERMINALS[g])
   else:
    ref=value['H_actual_terminal'][g];require(ref['path']==str(S/'receipts'/('migration-'+g+'-terminal.json')),'H wait names another job');t=pinned(ref);require(t['variant']==g and t['binding']==BINDING,'H migration wait lineage differs')
   require(t['actual_wait_returncode']==0,'Actual arm controller wait must be0')
  return dict(accepted=True,train_groups=4,completed_outers=100,old_supervisor_released=True,actual_K5_and_historical_G_lineage=True)
 replay.resource_boundary=boundary;replay.owned_release=release;replay.verify_four_group_terminal=terminal
 native_run=replay.run
 def run(*args):
  with Path('/home/shaoshitong/search-e1-config-study-20261004/experiment.lock').open('a+') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);return native_run(*args)
 replay.run=run
 return hf,common,replay

def main():
 parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--publication',required=True);parser.add_argument('--publication-sha256',required=True);parser.add_argument('--target-script',required=True);a,rest=parser.parse_known_args()
 target=Path(a.target_script).resolve();require(target.parent==ROOT and target.name in TARGETS,'Only own published analysis scripts')
 pubref=dict(path=a.publication,sha256=a.publication_sha256);setup(pubref);original=subprocess.Popen
 def dispatch(command,*args,**kwargs):
  if isinstance(command,(list,tuple)):
   command=list(command)
   if str(ROOT/'replay_worker.py') in command:
    i=command.index(str(ROOT/'replay_worker.py'));command=command[:i]+[str(ROOT/'entry.py'),'--publication',pubref['path'],'--publication-sha256',pubref['sha256'],'--target-script',str(ROOT/'replay_worker.py')]+command[i+1:]
  return original(command,*args,**kwargs)
 subprocess.Popen=dispatch;sys.argv=[str(target)]+rest
 module=importlib.import_module(target.stem);module.main()
if __name__=='__main__':main()
