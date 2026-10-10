"""K5 host/resource overlay over byte-frozen HF scientific evaluation."""
import argparse,copy,hashlib,importlib.util,json,os,subprocess,sys,types
from pathlib import Path
S=Path('/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010')
HF_SHA='ed5d7727f23fa6b8fd6b08c6d854f25fc1d31fb0bc2e9987c9b01daa3281d4d7'
BINDING_SHA='cdd237b50439d7caed5388f3d5cacad43d8fe0f978aa6b11c7c6c40e81bd2713'
SHAS={'G_POS':'e7c0659bc6544d3440b7f67b1ba2f976c56c79d8371e23b4dbe614cbb894f661','G_NEG':'2c37c851193dbc9a2d2512a72dd49310b9b8bcc14ac903159af4a491ef773a0d','H_POS':'934f3e1ab8f70d7a1faa14b9806e9ddaecdbaea6afc97668ca3389a5c7237612','H_NEG':'ddb7d18dfcd5826f6aede00bcb507cebe553d8af5a28b9ef3d299c137b92c8bc'}
def require(v,reason):
 if not v:raise ValueError(reason)
def pinned(ref):
 b=Path(ref['path']).read_bytes();require(hashlib.sha256(b).hexdigest()==ref['sha256'],'Pinned file changed');return json.loads(b)
def port_source(source):
 require(hashlib.sha256(source.encode()).hexdigest()==HF_SHA,'Frozen HF source differs')
 old="""    status=json.loads((S/'study-status.json').read_text());require(status['phase']=='FOUR_GROUPS_TRAIN_COMPLETE_TESTS_PENDING','All100 training outers must finish first')
    terminal=subprocess.check_output(['tmux','display-message','-p','-t','e1-signal-port-resume-0146350-20261010','#{pane_pid}|#{pane_dead}|#{pane_dead_status}'],text=True).strip()
    completed_training_terminal(status, read(status['receipt']), terminal)"""
 require(source.count(old)==1,'HF terminal patch ambiguous');source=source.replace(old,'    _actual_k5_training_terminal()')
 require(source.count('str(4+rank%4)')==1,'HF worker mapping changed');source=source.replace('str(4+rank%4)','str(rank%4)')
 # K5 CUDA destroy can take minutes: resource wait only, no sampling/scoring change.
 require(source.count('for retry in range(30):')==1,'HF release wait changed');source=source.replace('for retry in range(30):','for retry in range(300):')
 return source
WORKER_SHA='5ceec7847557d3ff34f09f82dfda0de7999bb230983af506ef63e689ef96f4c9'
def port_worker(source):
 require(hashlib.sha256(source.encode()).hexdigest()==WORKER_SHA,'Frozen HF worker source differs')
 require(source.count('str(4+rank%4)')==1 and source.count('physical_gpu=4+rank%4')==1,'HF worker mapping changed')
 return source.replace('str(4+rank%4)','str(rank%4)').replace('physical_gpu=4+rank%4','physical_gpu=rank%4')
def check_terminal_values(complete,terminals,ledgers,pane,server):
 require(complete['accepted'] and complete['training_outer_total']==100 and complete['groups']==list(SHAS),'All100 required')
 require(complete['binding']['sha256']==BINDING_SHA,'Training migration lineage differs')
 require(server==dict(pid=3535788,uid=1001,starttick=643725164),'Actual tmux server differs')
 require(pane=='3550866|1|0|$2|%2','Actual original foreground terminal0 required')
 for g in ['H_POS','H_NEG']:
  t=terminals[g];require(t['variant']==g and t['actual_wait_returncode']==0 and t['binding']['sha256']==BINDING_SHA,'Actual H child wait0 required')
 require(set(ledgers)==set(SHAS),'Four training ledgers required')
 for g,v in ledgers.items():
  require(v['accepted'] and v['outer_iterations']==25 and len(v['rows'])==25,'Each group needs25')
  require([r['outer_index'] for r in v['rows']]==list(range(25)),'Outer sequence differs')
  require(all(r['train_denominator']==2560 and 0<=r['train_correct']<=2560 for r in v['rows']),'All2560 TRAIN outcomes required')
 return True
def terminal():
 c=json.loads((S/'receipts/migration-training-complete.json').read_text())
 t={g:json.loads((S/'receipts'/('migration-'+g+'-terminal.json')).read_text()) for g in ['H_POS','H_NEG']}
 l={g:json.loads((S/('group-'+g)/'runs/training-em-25-complete.json').read_text()) for g in SHAS}
 pane=subprocess.check_output(['tmux','display-message','-p','-t','e1-signal-hkust5-isolated-r2-20261011','#{pane_pid}|#{pane_dead}|#{pane_dead_status}|#{session_id}|#{pane_id}'],text=True).strip()
 p=Path('/proc/3535788');stat=(p/'stat').read_text().rsplit(')',1)[1].split();server=dict(pid=3535788,uid=p.stat().st_uid,starttick=int(stat[19]))
 check_terminal_values(c,t,l,pane,server)
 for pid,tick in [(3550871,643848202),(3550876,643848220)]:
  p=Path('/proc',str(pid))
  if p.exists():require(int((p/'stat').read_text().rsplit(')',1)[1].split()[19])!=tick,'Original owned controller remains')
 return c
def setup(ref):
 require(ref['sha256']==BINDING_SHA,'Only the actual accepted migration')
 binding=pinned(ref);path=S/'source/hkust5-device-isolation-v2/opsd_signal_weight_hkust5_migration_v1/entry.py'
 spec=importlib.util.spec_from_file_location('_migration_original',path);migration=importlib.util.module_from_spec(spec);spec.loader.exec_module(migration)
 cp,prepare,gates=migration.install(binding)
 source=port_source((Path(__file__).parent/'evaluate.py').read_text())
 m=types.ModuleType('_k5_frozen_hf_resource_overlay');m.__file__=str(Path(__file__).resolve());m.__dict__['_actual_k5_training_terminal']=terminal
 exec(compile(source,str(Path(__file__).parent/'evaluate.py'),'exec'),m.__dict__)
 old_read=m.read
 def read(ref):
  v=old_read(ref)
  if v.get('schema')=='opsd-signal-weight-config-v1':
   require(ref['sha256']==SHAS[v['variant']],'Original arm config differs');v=copy.deepcopy(v);v['owner']=dict(uid=1001,physical_gpus=[0,1,2,3],gpu_uuids=migration.UUIDS[:])
  return v
 m.read=read
 # Frozen worker imports its matching parent API; do not import an old K4 parent.
 sys.modules['evaluate']=m
 w=types.ModuleType('_k5_frozen_hf_worker_resource_overlay');w.__file__=str(Path(__file__).parent/'worker.py')
 exec(compile(port_worker((Path(__file__).parent/'worker.py').read_text()),w.__file__,'exec'),w.__dict__)
 m.worker=w.run_worker
 def admission(config,allowed_run=None,start=True):
  devices,apps=migration.inventory();require([d['uuid'] for d in devices]==config['owner']['gpu_uuids'],'Actual K5 allocation differs')
  oracle=migration.oracle(config,gates)
  for app in apps:
   actual=gates.process(app['pid']);require(not start and actual['uid']==1001 and actual['run_id']==allowed_run,'Foreign/old GPU process; never signal')
  require(min(d['actual_free_mib'] for d in devices)>=(20480 if start else 6144),'Directfree below original floor')
  return dict(devices=devices,oracle=oracle,checked_unix=__import__('time').time())
 m.admission=admission
 native_run=m.run
 def run(*a):
  # Keep the target-only lease held by training plus all original HF leases.
  import fcntl
  with Path('/home/shaoshitong/search-e1-config-study-20261004/experiment.lock').open('a+') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);return native_run(*a)
 m.run=run
 return m
def main():
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--binding',required=True);p.add_argument('--binding-sha256',required=True);a,rest=p.parse_known_args()
 ref=dict(path=a.binding,sha256=a.binding_sha256);m=setup(ref);original=subprocess.Popen
 def dispatch(command,*args,**kwargs):
  if isinstance(command,(list,tuple)) and len(command)>1 and command[1]==str(Path(__file__).resolve()):
   command=list(command[:2])+['--binding',ref['path'],'--binding-sha256',ref['sha256']]+list(command[2:])
  return original(command,*args,**kwargs)
 subprocess.Popen=dispatch;sys.argv=[str(Path(__file__).resolve())]+rest;m.main()
if __name__=='__main__':main()
