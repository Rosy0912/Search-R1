import importlib.util,json,os
from pathlib import Path
import pytest
P=Path(__file__).with_name('capacity-tools')/'capacity_entry.py'
s=importlib.util.spec_from_file_location('capacity_test',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def snapshot():return {'inventory':[dict(index=i,uuid='GPU-'+str(i),name='H100',used=10,total=100,util=0) for i in range(8)],'applications':[]}
def test_strict_snapshot_empty_and_same8():m.validate_snapshot(snapshot(),idle=True)
@pytest.mark.parametrize('fault',['foreign','memory','util','duplicate'])
def test_snapshot_drift_rejected(fault):
 v=snapshot()
 if fault=='foreign':v['applications']=[dict(pid=999)]
 if fault=='memory':v['inventory'][0]['used']=16
 if fault=='util':v['inventory'][0]['util']=16
 if fault=='duplicate':v['inventory'][0]['uuid']='GPU-1'
 with pytest.raises(AssertionError):m.validate_snapshot(v,idle=True)
def resource_row():
 apps=[dict(pid=i+20,uuid='GPU-'+str(i),in_slurm_job=True,uid=str(os.getuid())) for i in range(8)]+[dict(pid=10,uuid='GPU-0',in_slurm_job=True,uid=str(os.getuid()))]
 return dict(job_id='100',time='2026-10-04T12:00:00+00:00',gpu_processes=apps)
def test_two_actual_gpu_samples(tmp_path):
 p=tmp_path/'samples';rows=[resource_row(),resource_row()];rows[1]['time']='2026-10-04T12:00:02+00:00';rows[1]['time']='2026-10-04T12:00:02+00:00';p.write_text('\n'.join(json.dumps(r) for r in rows))
 assert len(m.validate_resource_samples(p,10,['GPU-'+str(i) for i in range(8)],'100'))==2
@pytest.mark.parametrize('fault',['one','foreign','differentjob','changedpid','missinguuid'])
def test_resource_sample_falsepasses_rejected(tmp_path,fault):
 rows=[resource_row(),resource_row()];rows[1]['time']='2026-10-04T12:00:02+00:00'
 if fault=='one':rows.pop()
 if fault=='foreign':rows[0]['gpu_processes'][0]['in_slurm_job']=False
 if fault=='differentjob':rows[0]['job_id']='101'
 if fault=='changedpid':rows[1]['gpu_processes'][0]['pid']=999
 if fault=='missinguuid':rows[0]['gpu_processes'][0]['uuid']='foreign'
 p=tmp_path/'samples';p.write_text('\n'.join(json.dumps(v) for v in rows))
 with pytest.raises(ValueError):m.validate_resource_samples(p,10,['GPU-'+str(i) for i in range(8)],'100')
def test_duplicate_and_nonfinite_json_refused():
 for v in ['{"x":1,"x":2}','{"x":NaN}']:
  with pytest.raises(ValueError):m.unique(v)
def probe():
 rank=dict(passed=True,world_size=8,job_id='100',source_commit=m.SOURCE,local_micro=2,input_width=6144,response_width=4096,optimizer_steps=2,optimizer_step_records=[dict(loss=1.,grad_norm=1.)]*2,minimum_required_free_fraction=.15,estimated_peak_device_used_bytes=85,gpu_total_bytes=100,component_clip=None,teacher_prompt_mode='user_context',official_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',student_master_parameter_dtypes=['torch.float32'],student_mixed_precision_parameter_dtype='torch.bfloat16',teacher_active_runtime=dict(parameter_dtypes=['torch.float32'],attention='eager'),teacher_returned_runtime=dict(active_device='cpu'))
 return dict(passed=True,local_micro=2,ranks=[dict(rank,rank=i) for i in range(8)])
def test_probe_all8_twoadam_accepts():m.validate_probe(probe(),'100')
@pytest.mark.parametrize('fault',['missingrank','wrongjob','oneadam','oom','nan','clip','teacherbf16'])
def test_probe_falseaccept_rejected(fault):
 v=probe();r=v['ranks'][0]
 if fault=='missingrank':v['ranks'].pop()
 if fault=='wrongjob':r['job_id']='101'
 if fault=='oneadam':r['optimizer_steps']=1
 if fault=='oom':r['estimated_peak_device_used_bytes']=86
 if fault=='nan':r['optimizer_step_records']=[dict(loss=float('nan'),grad_norm=1.)]*2
 if fault=='clip':r['component_clip']=.05
 if fault=='teacherbf16':r['teacher_active_runtime']['parameter_dtypes']=['torch.bfloat16']
 with pytest.raises(AssertionError):m.validate_probe(v,'100')

@pytest.mark.parametrize('fault',['duplicate','backwards','naive','nonutc','invalid'])
def test_non_independent_resource_times_refused(tmp_path,fault):
 rows=[resource_row(),resource_row()]
 rows[1]['time']={'duplicate':rows[0]['time'],'backwards':'2026-10-04T11:59:59+00:00','naive':'2026-10-04T12:00:02','nonutc':'2026-10-04T20:00:02+08:00','invalid':'now'}[fault]
 p=tmp_path/'samples';p.write_text('\n'.join(json.dumps(r) for r in rows))
 with pytest.raises(ValueError):m.validate_resource_samples(p,10,['GPU-'+str(i) for i in range(8)],'100')
