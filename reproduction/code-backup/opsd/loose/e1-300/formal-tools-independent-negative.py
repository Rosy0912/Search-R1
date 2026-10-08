"""Independent actual wrapper-function CPU counterexamples, no source edits/GPU."""
import importlib.util,json,os,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent;DIR=ROOT/'formal-tools';sys.path.insert(0,str(DIR))
spec=importlib.util.spec_from_file_location('formal_independent',DIR/'formal_supervisor.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
results={}
def rejects(name,fn):
 try:fn()
 except (ValueError,AssertionError,RuntimeError):results[name]='REJECTED_EXPECTED';return
 raise AssertionError('Independent negative passed: '+name)
with tempfile.TemporaryDirectory() as td:
 p=Path(td)/'metadata.json';p.write_text('{"value":1}');old=f.sha(p);p.write_text('{"value":2}')
 rejects('joint_current_metadata_cannot_replace_external_pin',lambda:f.pinned(p,old))
 inv={'inventory':[dict(index=i,uuid='GPU-'+str(i),name='H100',used=1.,total=100.,util=0.) for i in range(8)],'applications':[dict(pid=17,uuid='GPU-'+str(i)) for i in range(8)]}
 f.validate_snapshot(inv,[r['uuid'] for r in inv['inventory']],17,idle=True)
 rejects('actual_service_pid_mismatch',lambda:f.validate_snapshot(inv,[r['uuid'] for r in inv['inventory']],18,idle=True))
 row={'job_id':'CPU123','time':'2026-10-04T00:00:00+00:00','gpus':[dict(uuid='GPU-'+str(i),util=0.,used_mib=1.,total_mib=100.) for i in range(8)],'gpu_processes':[dict(pid=17,uuid='GPU-'+str(i),uid=str(os.getuid()),in_slurm_job=True) for i in range(8)]+[dict(pid=30+i,uuid='GPU-'+str(i),uid=str(os.getuid()),in_slurm_job=True) for i in range(8)]}
 path=Path(td)/'resources'
 def write(rows):path.write_text('\n'.join(json.dumps(r) for r in rows))
 write([row,row]);rejects('duplicate_sampler_record',lambda:f.validate_resources(path,'CPU123',['GPU-'+str(i) for i in range(8)],17))
 second=json.loads(json.dumps(row));second['time']='2026-10-04T00:00:02+00:00';second['gpu_processes'][0]['in_slurm_job']=False;write([row,second]);rejects('foreign_GPU_application',lambda:f.validate_resources(path,'CPU123',['GPU-'+str(i) for i in range(8)],17))
 bad=dict(row,job_id='CPUold');write([row,bad]);rejects('cross_job_resource_reuse',lambda:f.validate_resources(path,'CPU123',['GPU-'+str(i) for i in range(8)],17))
 print(json.dumps({'scope':'Independent CPU counterexamples only','cases':results,'passed':len(results)==5},sort_keys=True))
