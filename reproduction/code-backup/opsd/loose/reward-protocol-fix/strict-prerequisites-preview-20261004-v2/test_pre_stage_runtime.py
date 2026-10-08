"""Synthetic CPU schema fixtures; do not claim actual GPU ownership/activation."""
import copy,json,shutil
from pathlib import Path
import pytest
from pre_stage_runtime import verify_hf_activation,verify_resource_rows,verify_pre_stage,HF_TEMPLATE_SHA
from training_evidence import digest,COMMIT
HERE=Path(__file__).parent

def activation(tmp_path):
 folder=tmp_path/'hf-final-review-fixture';folder.mkdir()
 original=json.loads((HERE/'source-manifest.json').read_text())
 for name in original['file_sources']:shutil.copyfile(HERE/name,folder/name)
 p=folder/'reload_job.sh';guard='echo "STAGED PREVIEW: independent review required" >&2\nexit 78\n';p.write_text(p.read_text().replace(guard,'').replace(original['directory_name'],folder.name))
 actual=copy.deepcopy(original);actual['preview_blocked']=False;actual['directory_name']=folder.name;actual['prior_hf_preview_manifest_sha256']=HF_TEMPLATE_SHA
 actual['file_sources']['reload_job.sh']=digest(p)
 m=folder/'source-manifest.json';m.write_text(json.dumps(actual));return folder,m,actual

def test_actual_final_manifest_can_differ_from_blocked_preview(tmp_path):
 folder,m,actual=activation(tmp_path)
 assert digest(m)!=digest(HERE/'source-manifest.json')
 verify_hf_activation(m,folder/'reload_job.sh',folder/'checkpoint_reload.py',digest(m))

@pytest.mark.parametrize('mutation',['guard','ancestry','checker','directory','extra','hash'])
def test_activation_rejects_guard_unknown_edits_or_unbound_source(tmp_path,mutation):
 folder,m,actual=activation(tmp_path)
 if mutation=='guard':shutil.copyfile(HERE/'reload_job.sh',folder/'reload_job.sh');actual['file_sources']['reload_job.sh']=digest(folder/'reload_job.sh')
 if mutation=='ancestry':actual['prior_hf_preview_manifest_sha256']='0'*64
 if mutation=='checker':p=folder/'checkpoint_reload.py';p.write_text(p.read_text()+'\nprint("fake")\n');actual['file_sources']['checkpoint_reload.py']=digest(p)
 if mutation=='directory':actual['directory_name']='other'
 if mutation=='extra':actual['file_sources']['fake.py']='0'*64
 if mutation=='hash':actual['file_sources']['reload_job.sh']='0'*64
 m.write_text(json.dumps(actual))
 with pytest.raises(ValueError):verify_hf_activation(m,folder/'reload_job.sh',folder/'checkpoint_reload.py',digest(m))

def resources():
 p=dict(pid=17,uid='1000',process_start_ticks='123',in_slurm_job=True)
 row=dict(time='2026-10-04T00:00:00+00:00',job_id='CPU_SCHEMA_FIXTURE',nominal_sleep_seconds=2,gpus=[dict(uuid='GPU-fixture',index='0',util=1,used_mib=20,total_mib=100)],gpu_processes=[dict(p,uuid='GPU-fixture',gpu_used_mib='20')],job_processes=[p],cgroup={'/sys/fs/cgroup/slurm/job_CPU_SCHEMA_FIXTURE/memory.current':'123'})
 return [row,dict(copy.deepcopy(row),time='2026-10-04T00:00:02+00:00')]

def test_cpu_resource_schema_only():
 assert verify_resource_rows(resources(),'CPU_SCHEMA_FIXTURE','GPU-fixture','1000',17,'123')['samples']==2

@pytest.mark.parametrize('mutation',['job','time','query','uuid','pid','uid','foreign','start','cgroup','activity','nan'])
def test_resource_ownership_and_identity_negatives(mutation):
 rows=resources();r=rows[0]
 if mutation=='job':r['job_id']='other'
 if mutation=='time':rows[1]['time']=r['time']
 if mutation=='query':r['query_error']='Timeout'
 if mutation=='uuid':r['gpus'][0]['uuid']='other'
 if mutation=='pid':r['gpu_processes'][0]['pid']=18
 if mutation=='uid':r['gpu_processes'][0]['uid']='1001'
 if mutation=='foreign':r['gpu_processes'][0]['in_slurm_job']=False
 if mutation=='start':r['gpu_processes'][0]['process_start_ticks']='124'
 if mutation=='cgroup':r['cgroup']={'/sys/fs/cgroup/slurm/job_other/memory.current':'123'}
 if mutation=='activity':
  for row in rows:row['gpu_processes']=[]
 if mutation=='nan':r['gpus'][0]['used_mib']=float('nan')
 with pytest.raises(ValueError):verify_resource_rows(rows,'CPU_SCHEMA_FIXTURE','GPU-fixture','1000',17,'123')

def test_future_eval_completion_is_not_pre_stage_schema():
 with pytest.raises(ValueError,match='pre-generation'):verify_pre_stage({'version':1,'phase':'post-generation','corrected_eval_job':'UNFILLED'},{})

def test_source_import_capture_requires_actual_modules(tmp_path):
 from runtime_capture import imported_source_records
 from types import SimpleNamespace
 p=tmp_path/'actual.py';p.write_text('# CPU fixture source')
 assert imported_source_records(['actual'],tmp_path,{'actual':SimpleNamespace(__file__=str(p))})['actual']['sha256']==digest(p)
 with pytest.raises(ValueError):imported_source_records(['missing'],tmp_path,{})
 outside=tmp_path/'sub';outside.mkdir()
 with pytest.raises(ValueError):imported_source_records(['actual'],outside,{'actual':SimpleNamespace(__file__=str(p))})

def test_checker_identity_producer_has_no_execution_side_effects(tmp_path):
 from runtime_capture import checker_identity
 proc=tmp_path/'proc';proc.mkdir();checker=tmp_path/'checkpoint_reload.py';checker.write_text('# immutable CPU fixture')
 def record(pid,ppid,cmd,job='CPU_SCHEMA_FIXTURE'):
  p=proc/str(pid);p.mkdir();fields=['S',str(ppid)]+['0']*18;fields[19]='123';(p/'stat').write_text(f'{pid} (python) '+' '.join(fields));(p/'cmdline').write_bytes(b'\0'.join(x.encode() for x in cmd)+b'\0');(p/'cgroup').write_text('0::/slurm/job_'+job+'/step_batch\n')
 record(10,1,['python','preflight.py']);record(11,10,['python',str(checker.resolve())]);proof=checker_identity(10,checker,'CPU_SCHEMA_FIXTURE',proc)
 assert proof['checker']['pid']==11 and proof['checker_sha256']==digest(checker)
 record(12,10,['python',str(checker.resolve())])
 with pytest.raises(ValueError,match='Exactly one'):checker_identity(10,checker,'CPU_SCHEMA_FIXTURE',proc)
