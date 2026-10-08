"""CPU synthetic schema tests only; not actual corrected GPU/job acceptance."""
import copy,hashlib,os,subprocess,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parent))
import post_eval_check as post
from runtime_capture import process_identity

def external():
 return {'version':1,'phase':'post-evaluation-external-binding','job_id':'101','uid':os.getuid(),'eval_repo':'/tmp/CPU-schema-only-e3','eval_commit':post.EVAL_COMMIT,'stage_sha256':'a'*64,'input_manifest_sha256':'b'*64,'model_role':'base','selected_gpu_uuids':['GPU-'+f'{i:08x}'+'-0000-0000-0000-000000000000' for i in range(8)],'post_verifier_sha256':'c'*64,'launcher_bundle_sha256':'d'*64,'canonical_template_manifest':{'path':'/tmp/CPU-only-template/source-manifest.json','sha256':'e'*64}}

def rows_and_transcript():
 ext=external();stage={'source_files':{'fixture.py':'a'*64}};workers=[]
 # Explicit synthetic Linux /proc schema; no actual Slurm/GPU proof is constructed.
 fields=['S','999']+['0']*18;fields[19]='1234'
 uid=ext['uid'];cmd=b'python\0cpu-schema-fixture\0'
 actual={'pid':1000,'ppid':999,'process_start_ticks':'1234','uid':uid,'uids':[uid]*4,'cgroup':'/slurm/job_101/step_0','cmdline':['python','cpu-schema-fixture'],'raw_stat':'1000 (cpu-fixture) '+' '.join(fields),'raw_status':'Uid:\t'+'\t'.join([str(uid)]*4),'raw_cgroup':'/slurm/job_101/step_0','raw_cmdline_hex':cmd.hex()}
 for i in range(8):
  pid=1000+i;process=copy.deepcopy(actual);process.update(pid=pid,cgroup='/slurm/job_101/step_0',raw_cgroup='/slurm/job_101/step_0');process['raw_stat']=str(pid)+' '+actual['raw_stat'].split(' ',1)[1]
  workers.append({'rank':i,'pid':pid,'uid':ext['uid'],'node':'h100-10','job_id':'101','stage_sha256':'a'*64,'gpu_uuid':ext['selected_gpu_uuids'][i],'device_index':i,'process':process})
 spec={key:ext[key] for key in ('eval_repo','job_id','uid')};spec.update(source_files=stage['source_files'],stage_sha256=ext['stage_sha256'])
 transcript={'count':51713,'files':[{'count':64} for _ in range(808)]+[{'count':1}],'zero_update_guard':{'attempted_update_dispatches':0,'guarded_dispatches':['actor_rollout_wg.update_actor'],'corrected_runtime':{'rejected_calls':0,'worker_audit':{'spec':spec,'rows':workers},'generation_calls':[{'sequence':1,'rows':64,'do_sample':False,'validate':True}]}}}
 return ext,stage,transcript

def test_exact_external_schema():post.exact_external(external())
@pytest.mark.parametrize('key,value',[('phase','pre-generation'),('job_id','UNFILLED'),('job_id','3007'),('uid','1000'),('eval_commit','0'*40),('model_role','teacher'),('selected_gpu_uuids',['fake']*8),('stage_sha256','UNFILLED')])
def test_external_selflabels_reject(key,value):
 ext=external();ext[key]=value
 with pytest.raises(ValueError):post.exact_external(ext)

def test_future_receipt_or_backrefs_reject():
 ext=external();ext['future_receipt']=True
 with pytest.raises(ValueError):post.exact_external(ext)

def test_full_native_batch_and_external_worker_schema():
 ext,stage,transcript=rows_and_transcript();assert len(post.runtime_crossbinding(transcript,stage,ext))==8

@pytest.mark.parametrize('mutation',['job','repo','uid','stage','foreign_uid','duplicate_rank','duplicate_uuid','invalid_device','batch64','missing_tail','extra_batch','update','sample','validate'])
def test_external_worker_and_native_batch_tamper_reject(mutation):
 ext,stage,t=rows_and_transcript();runtime=t['zero_update_guard']['corrected_runtime'];spec=runtime['worker_audit']['spec'];rows=runtime['worker_audit']['rows']
 if mutation in ('job','repo','uid','stage'):spec[{'job':'job_id','repo':'eval_repo','uid':'uid','stage':'stage_sha256'}[mutation]]='changed'
 elif mutation=='foreign_uid':rows[0]['process']['uids'][2]+=1
 elif mutation=='duplicate_rank':rows[0]['rank']=1
 elif mutation=='duplicate_uuid':rows[0]['gpu_uuid']=rows[1]['gpu_uuid']
 elif mutation=='invalid_device':rows[0]['device_index']=.5
 elif mutation=='batch64':t['files'][-1]['count']=64
 elif mutation=='missing_tail':t['files'].pop()
 elif mutation=='extra_batch':t['files'].append({'count':1})
 elif mutation=='update':t['zero_update_guard']['attempted_update_dispatches']=1
 elif mutation=='sample':runtime['generation_calls'][0]['do_sample']=True
 elif mutation=='validate':runtime['generation_calls'][0]['validate']=False
 with pytest.raises(ValueError):post.runtime_crossbinding(t,stage,ext)

def test_actual_scheduler_user_and_normal_terminal_schema():
 ext=external();text='JobId=101 JobState=COMPLETED ExitCode=0:0 NodeList=h100-10 NumNodes=1 TresPerNode=gres:gpu:h100:8 UserId=fixture('+str(ext['uid'])+')'
 # Synthetic terminal TEXT schema only, never supplied to verify_receipt.
 post.normal_terminal(text,ext)
 for bad in [text.replace('0:0','0:15'),text.replace('COMPLETED','RUNNING'),text.replace('h100-10','a800'),text+' JobId=101',text.replace('('+str(ext['uid'])+')','(9999999)')]:
  with pytest.raises((ValueError,AssertionError)):post.normal_terminal(bad,ext)

def metrics_proof(role,job):
 return {'kind':'actual-corrected-final-evaluation','model_role':role,'job_id':job,'input_manifest_sha256':'a'*64,'generation_budget_sha256':'b'*64,'source_files':{'scorer.py':'c'*64},'eval_commit':post.EVAL_COMMIT,'questions':51713,'batches':809,'metrics':{'val/em/all':.5},'receipt_sha256':'d'*64}

def test_matching_pair_schema():assert post.compare_pair(metrics_proof('base','101'),metrics_proof('hf500','102'))['delta']=={'val/em/all':0.0}
@pytest.mark.parametrize('key,value',[('model_role','base'),('job_id','101'),('input_manifest_sha256','e'*64),('generation_budget_sha256','e'*64),('source_files',{}),('eval_commit','0'*40),('questions',64),('batches',1),('metrics',{})])
def test_pair_mismatched_input_budget_source_role_scope_reject(key,value):
 hf=metrics_proof('hf500','102');hf[key]=value
 with pytest.raises(ValueError):post.compare_pair(metrics_proof('base','101'),hf)

@pytest.mark.parametrize('filename',['post_eval_check.py','eval_supervisor.py','eval_job.py','retrieval_audit_entry.py'])
def test_future_entries_are_blocked(filename):
 result=subprocess.run([sys.executable,str(Path(__file__).parent/filename)],capture_output=True,text=True)
 assert result.returncode==78 and 'BLOCKED' in result.stderr

def resource_fixture():
 ext,stage,t=rows_and_transcript();workers=t['zero_update_guard']['corrected_runtime']['worker_audit']['rows'];retriever={'process':copy.deepcopy(workers[0]['process'])};retriever['process']['pid']=2000
 identities=workers+[{'pid':2000,'process':retriever['process']}]
 own=[{'pid':r['pid'],'uid':str(ext['uid']),'process_start_ticks':'1234','in_slurm_job':True} for r in identities]
 gpu=[{'pid':r['pid'],'uuid':r['gpu_uuid'],'uid':str(ext['uid']),'process_start_ticks':'1234','in_slurm_job':True,'gpu_used_mib':'100'} for r in workers]
 gpu.extend({'pid':2000,'uuid':u,'uid':str(ext['uid']),'process_start_ticks':'1234','in_slurm_job':True,'gpu_used_mib':'100'} for u in ext['selected_gpu_uuids'])
 rows=[{'time':'2026-10-04T00:00:00+00:00','job_id':'101','gpus':[{'uuid':u,'util':2.,'used_mib':200.,'total_mib':80000.} for u in ext['selected_gpu_uuids']],'gpu_processes':gpu,'job_processes':own,'cgroup':{'/sys/fs/cgroup/slurm/job_101/memory.current':'1000','/sys/fs/cgroup/slurm/job_101/memory.events':'oom 0\noom_kill 0'}}]
 return ext,workers,retriever,rows

def test_actual_frozen_sampler_schema_not_imagined_fields():
 ext,workers,retriever,rows=resource_fixture();post.resources_crossbinding(rows,workers,retriever,ext)

@pytest.mark.parametrize('mutation',['empty','time','job','GPU','foreign_uid','pid_reused','gpu_nan','gpu_negative','cgroup','events','missing_worker','missing_retriever'])
def test_resource_ownership_gpu_and_cgroup_tamper_reject(mutation):
 ext,workers,retriever,rows=resource_fixture();row=rows[0]
 if mutation=='empty':rows=[]
 elif mutation=='time':row['time']='not ISO actual time'
 elif mutation=='job':row['job_id']='102'
 elif mutation=='GPU':row['gpus'][0]['uuid']='other'
 elif mutation=='foreign_uid':row['gpu_processes'][0]['uid']='9999999'
 elif mutation=='pid_reused':row['gpu_processes'][0]['process_start_ticks']='99999'
 elif mutation=='gpu_nan':row['gpu_processes'][0]['gpu_used_mib']='NaN'
 elif mutation=='gpu_negative':row['gpu_processes'][0]['gpu_used_mib']='-1'
 elif mutation=='cgroup':row['cgroup']={'/sys/fs/cgroup/slurm/job_102/memory.current':'1000'}
 elif mutation=='events':row['cgroup']['/sys/fs/cgroup/slurm/job_101/memory.events']='oom invalid'
 elif mutation=='missing_worker':row['gpu_processes']=row['gpu_processes'][1:]
 elif mutation=='missing_retriever':row['gpu_processes']=row['gpu_processes'][:8]
 with pytest.raises(ValueError):post.resources_crossbinding(rows,workers,retriever,ext)
