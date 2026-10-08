"""Small raw contracts only; no real weight scan or model proof."""
import importlib.util,pathlib,json,hashlib,pytest,copy
p=pathlib.Path(__file__).with_name('admission.py');s=importlib.util.spec_from_file_location('scan_admission',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def setup(tmp,monkeypatch):
 monkeypatch.setattr(m,'ROOT',tmp);(tmp/'logs').mkdir();(tmp/'proc').mkdir()
 def save(n,x):
  p=tmp/'logs'/n;p.write_text(json.dumps(x));return {'path':str(p),'sha256':m.sha(p)}
 cleanup={'cleanup':{'actual_cgroup_empty':True},'terminal_monitors_drained_before_authorized_exit':True,'parent_cuda_initialized':False,'error':None,'primary_error':None,'identities':{'90':{'starttime_ticks':100}}}
 for k in ('own_processes_remaining','own_gpu_records_remaining','parent_gpu_records','baseline_identity_missing','original_gpu_baseline_missing','guardian_errors','sampler_errors','failure_history'):cleanup[k]=[]
 outer={'actual_completed_outer_steps':[1,2],'export_step':2,'checkpoint_directory':str(tmp/'model')}
 source=tmp/'logs/source.json';source.write_bytes(pathlib.Path(__file__).parent.parent.joinpath('search-r1-32b-grpo100-source-v4/r1-files.json').read_bytes())
 config={'job':'99','mode':'profile2','expected_outer_steps':2,'final_checkpoint':str(tmp/'model'),'source_manifest':{'path':str(source),'sha256':m.SOURCE},'tools_manifest':save('tools.json',{})}
 q={'scope':'REQUEST_BOUNDED_CPU_HF_SEMANTIC_SCAN_NOT_APPROVAL','mode':'profile2','job_id':99,'review_tool_manifest':{'path':str(tmp/'tools/r1-32b-hf-semantic-review-v5/files.json'),'sha256':m.MANIFEST},'effective_config':save('config.json',config),'controller_terminal':save('terminal.json',{'job_id':99,'returncode':0,'stdout':'JobId=99 JobState=COMPLETED ExitCode=0:0'})}
 result={'scope':'ACTUAL_SEARCH_R1_PROFILE2_UNREVIEWED_NOT_100_NOT_TOPD','job':'99','status':'COMPLETE_UNREVIEWED','cleanup_passed':True,'error':None,'effective_config_sha256':q['effective_config']['sha256'],'actual_outer_evidence':save('outer.json',outer),'export':{'directory':str(tmp/'model'),'tensor_count':771,'shards':['p.safetensors'],'files':{'p.safetensors':{'size':1,'sha256':'x'}}},'cleanup':save('cleanup.json',cleanup)}
 q['train_result']=save('result.json',result);return q,result,cleanup,save

def test_complete_metadata_allowed_for_later_original_scan(tmp_path,monkeypatch):
 q,*_=setup(tmp_path,monkeypatch);m.check_request(q,tmp_path/'proc')
@pytest.mark.parametrize('case',['failed','live','profile_as100','terminalFAILED','missing_outer','cleanup','source','incompleteHF'])
def test_failed_incomplete_or_identity_drift_refused(tmp_path,monkeypatch,case):
 q,r,cl,save=setup(tmp_path,monkeypatch)
 if case=='failed':r['status']='FAILED';r['error']='original failure'
 if case=='live':r['cleanup_passed']=False
 if case=='profile_as100':q['mode']='train100'
 if case=='terminalFAILED':q['controller_terminal']=save('terminal2.json',{'job_id':99,'returncode':0,'stdout':'JobId=99 JobState=FAILED ExitCode=1:0'})
 if case=='missing_outer':r['actual_outer_evidence']=save('outer2.json',{'actual_completed_outer_steps':[1],'export_step':2})
 if case=='cleanup':cl['guardian_errors']=['unknown'];r['cleanup']=save('cleanup2.json',cl)
 if case=='source':c={'job':'99','mode':'profile2','expected_outer_steps':2,'final_checkpoint':str(tmp_path/'model'),'source_manifest':{'sha256':'0'*64}};q['effective_config']=save('config2.json',c);r['effective_config_sha256']=q['effective_config']['sha256']
 if case=='incompleteHF':r['export']['tensor_count']=770
 q['train_result']=save('result2.json',r)
 with pytest.raises(ValueError):m.check_request(q,tmp_path/'proc')
def test_fresh_actual_old_job_cgroup_refused(tmp_path,monkeypatch):
 q,*_=setup(tmp_path,monkeypatch);p=tmp_path/'proc/90';p.mkdir();(p/'cgroup').write_text('1:freezer:/slurm/job_99/step_batch\n')
 with pytest.raises(ValueError,match='live cgroup'):m.check_request(q,tmp_path/'proc')
def test_fresh_old_start_refused_but_reused_pid_not_owned(tmp_path,monkeypatch):
 q,*_=setup(tmp_path,monkeypatch);p=tmp_path/'proc/90';p.mkdir();(p/'cgroup').write_text('1:freezer:/else\n');(p/'stat').write_text('90 (own) '+' '.join(['S']+['0']*18+['100']))
 with pytest.raises(ValueError,match='start still alive'):m.check_request(q,tmp_path/'proc')
 (p/'stat').write_text('90 (other) '+' '.join(['S']+['0']*18+['101']));m.check_request(q,tmp_path/'proc')
def test_original_raw_hash_drift_fails(tmp_path,monkeypatch):
 q,*_=setup(tmp_path,monkeypatch);pathlib.Path(q['train_result']['path']).write_text('{}')
 with pytest.raises(ValueError,match='raw SHA'):m.check_request(q,tmp_path/'proc')
