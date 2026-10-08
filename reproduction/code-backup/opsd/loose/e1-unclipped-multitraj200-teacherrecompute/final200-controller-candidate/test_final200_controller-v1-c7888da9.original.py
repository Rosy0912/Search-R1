"""Explicit local CPU software fixtures. Never production start/Slurm/GPU."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fixture_final200_controller',HERE/'final200_cpu_once.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
SEALED=HERE.parent/'final200-candidate'
raw=(SEALED/'common.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()==m.COMMON
c=types.ModuleType('fixture_sealed_common');c.__file__=str(SEALED/'common.py');exec(compile(raw,c.__file__,'exec',dont_inherit=True),c.__dict__)


def fixture_identity(pid):
 return dict(pid=pid,start_ticks=123456,uid=0,uids=[0]*4,state='S',node='h100-10',cgroup='1:freezer:/slurm_h100-10/uid_0/job_3304/step_3\n')


def test_exact_current_sealed_pins():
 for name,pin in m.PINS.items():assert hashlib.sha256((SEALED/name).read_bytes()).hexdigest()==pin


def test_reservation_unknown_never_retried(tmp_path):
 out=tmp_path/'unknown-once';m.reserve_once(out)
 m.write_once(out/'partial.original.json',{'scope':'SYNTHETIC_LOCAL_CPU_FIXTURE_NOT_ACTUAL200','waited':False})
 with pytest.raises(ValueError,match='never retried'):m.reserve_once(out)
 assert json.loads((out/'partial.original.json').read_text())['waited'] is False


def test_directory_reservation_fsync_before_files(tmp_path,monkeypatch):
 observed=[];real=m.sync_dir
 monkeypatch.setattr(m,'sync_dir',lambda p:(observed.append(Path(p)),real(p)))
 out=tmp_path/'once';m.reserve_once(out)
 assert observed==[tmp_path,out] and list(out.iterdir())==[]


def test_exclusive_output_no_overwrite(tmp_path):
 p=tmp_path/'original.json';m.write_once(p,{'actual_exit_fixture':7})
 with pytest.raises(FileExistsError):m.write_once(p,{'actual_exit_fixture':0})
 assert json.loads(p.read_text())['actual_exit_fixture']==7


def test_output_symlink_refused(tmp_path):
 original=tmp_path/'target';original.write_text('keep');alias=tmp_path/'alias';alias.symlink_to(original)
 with pytest.raises(FileExistsError):m.write_once(alias,{'fake':True})
 assert original.read_text()=='keep'


def test_stable_reader_symlink_and_hash(tmp_path):
 p=tmp_path/'input';p.write_bytes(b'exact');pin=m.sha(b'exact')
 assert m.stable_raw(p,pin)==b'exact'
 with pytest.raises(ValueError):m.stable_raw(p,'0'*64)
 alias=tmp_path/'alias';alias.symlink_to(p)
 with pytest.raises(ValueError):m.stable_raw(alias)
 with pytest.raises(ValueError):m.stable_raw(p,pin,limit=4)


def test_same_fd_mutation_during_read_rejected(tmp_path,monkeypatch):
 p=tmp_path/'read-race';p.write_bytes(b'original');real=os.read;mutated=[]
 def read(fd,size):
  raw=real(fd,size)
  if raw and not mutated:p.write_bytes(b'replaced');mutated.append(True)
  return raw
 monkeypatch.setattr(m.os,'read',read)
 with pytest.raises(ValueError,match='changed'):m.stable_raw(p)


@pytest.fixture
def local_bundle(tmp_path,monkeypatch):
 """Explicit locally packaged software fixture, never remote deployment."""
 root=tmp_path/'synthetic-root';(root/'tools').mkdir(parents=True)
 members={name:(SEALED/name).read_bytes() for name in m.PINS}
 members['final200_cpu_once.py']=(HERE/'final200_cpu_once.py').read_bytes()
 manifest=dict(version=1,kind='FINAL200_READ_ONLY_RUNTIME_BUNDLE',source_commit=m.SOURCE,tools={n:m.sha(v) for n,v in members.items()})
 encoded=m.canonical(manifest);expected=m.sha(encoded);directory=root/'tools'/('final200-'+expected[:12]);directory.mkdir()
 for name,raw in members.items():(directory/name).write_bytes(raw);(directory/name).chmod(0o444)
 (directory/'manifest.json').write_bytes(encoded);(directory/'manifest.json').chmod(0o444);directory.chmod(0o555)
 math_helper=tmp_path/'math_fixture.py';math_helper.write_bytes(b'# Synthetic unexecuted math dependency fixture\n')
 hf_helper=tmp_path/'hf_fixture.py';hf_helper.write_bytes(b'# Synthetic unexecuted HF dependency fixture\n')
 monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(m,'HERE',directory/'final200_cpu_once.py');monkeypatch.setattr(m,'host',lambda:None)
 monkeypatch.setattr(m,'MATH',math_helper);monkeypatch.setattr(m,'MATH_SHA',m.sha(math_helper.read_bytes()))
 monkeypatch.setattr(m,'HF',hf_helper);monkeypatch.setattr(m,'HF_SHA',m.sha(hf_helper.read_bytes()))
 yield directory,expected,manifest['tools']['final200_cpu_once.py']
 directory.chmod(0o700)
 for p in directory.iterdir():
  if not p.is_symlink():p.chmod(0o600)


def test_bundle_exact_source_compilation_local_fixture(local_bundle):
 directory,pin,own=local_bundle;result,common=m.bundle(pin,own)
 assert result==directory and common.CONFIG==m.CONFIG and common.__file__==str(directory/'common.py')


def test_common_changed_before_exec_rejected(local_bundle,tmp_path):
 directory,pin,own=local_bundle;source=directory/'common.py';marker=tmp_path/'forbidden-exec';source.chmod(0o600)
 source.write_bytes(source.read_bytes()+('\nfrom pathlib import Path\nPath('+repr(str(marker))+').write_text("BAD")\n').encode());source.chmod(0o444)
 with pytest.raises(ValueError,match='SHA differs'):m.bundle(pin,own)
 assert not marker.exists()


def test_bundle_extra_and_symlink_members_refused(local_bundle):
 directory,pin,own=local_bundle;directory.chmod(0o700);extra=directory/'unexpected';extra.write_text('unmanifested')
 with pytest.raises(ValueError,match='Unmanifested'):m.bundle(pin,own)
 extra.unlink();source=directory/'common.py';source.unlink();source.symlink_to(SEALED/'common.py');directory.chmod(0o555)
 with pytest.raises(ValueError,match='Physical'):m.bundle(pin,own)


@pytest.mark.parametrize('raw',[b'{"a":1,"a":2}',b'{"x":NaN}',b'{"x":Infinity}',b'{"x":1e999}'])
def test_json_unknown_values_rejected(raw):
 with pytest.raises(ValueError):m.parse_json(raw)


def test_srun_cpu_only_explicit_environment_and_cwd():
 argv=m.srun_command(Path('/actual/once-intent.original.json'),'a'*64,'3304')
 assert argv[:2]==['srun','--jobid=3304']
 for value in ['--overlap','--gres=none','--cpus-per-task=4','--nodelist=h100-10','--chdir='+str(m.REPO),'CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','PYTHONPATH='+str(m.REPO)]:assert value in argv
 assert '--mem=128G' not in argv and argv[-4:]==['--launch','/actual/once-intent.original.json','--launch-sha256','a'*64]
 with pytest.raises(ValueError):m.srun_command(Path('/unknown'),'a'*64,'901')


def test_helper_order_inputs_only_actual_report_shas(tmp_path):
 refs={'observe':{'sha256':'b'*64},'native':{'sha256':'c'*64}}
 commands=m.helper_commands(SEALED,tmp_path,'3304','3',refs)
 assert list(commands)==['observe','native','hf','archives']
 assert '--inventory-sha256' not in commands['observe']
 for stage in ('native','hf','archives'):
  a=commands[stage];assert a[a.index('--inventory-sha256')+1]=='b'*64
 for stage in ('hf','archives'):
  a=commands[stage];assert a[a.index('--native-report-sha256')+1]=='c'*64
  assert a[a.index('--native-inspector-sha256')+1]==m.PINS['inspect_final200_native.py']
 assert commands['native'][commands['native'].index('--math-helper')+1]==str(m.MATH)
 assert commands['hf'][commands['hf'].index('--hf-helper')+1]==str(m.HF)
 assert '2000' not in sum(commands.values(),[])  # clocks belong to content, not fabricated CLI values.


def test_wait_actual_child_failure_keeps_bytes_and_code(tmp_path):
 argv=[sys.executable,'-B','-c','import sys;print("original stdout");print("original stderr",file=sys.stderr);sys.exit(7)']
 result,start=m.wait_child(argv,tmp_path,'failed',fixture_identity)
 assert result['waited'] is True and result['returncode']==7 and result['child_pid']==start['child_pid']
 assert (tmp_path/'failed-stdout.original').read_text()=='original stdout\n'
 assert (tmp_path/'failed-stderr.original').read_text()=='original stderr\n'
 saved=json.loads((tmp_path/'failed-exit.original.json').read_text());assert saved['returncode']==7 and saved['waited_by_pid']==os.getpid()


def test_wait_even_if_identity_capture_fails(tmp_path):
 def rejected(pid):raise RuntimeError('Explicit synthetic identity-read failure')
 result,start=m.wait_child([sys.executable,'-c','raise SystemExit(6)'],tmp_path,'metadata',rejected)
 assert result['returncode']==6 and result['waited'] is True and len(result['capture_errors'])==2
 assert start['child'] is None and json.loads((tmp_path/'metadata-exit.original.json').read_text())['returncode']==6


def test_spawn_failure_wait_false_preserved(tmp_path):
 with pytest.raises(FileNotFoundError):m.wait_child(['/explicit/synthetic/no-such-python'],tmp_path,'spawn',fixture_identity)
 result=json.loads((tmp_path/'spawn-spawn-failure.original.json').read_text())
 assert result['waited'] is False and not (tmp_path/'spawn-exit.original.json').exists()


def fixture_pipeline(tmp_path,failed=None,bad_content=None,final_fail=False):
 out=tmp_path/'fixture-once';m.reserve_once(out);calls=[]
 filenames={'observe':'observed200.original.json','native':'native200.original.json','hf':'hf200.original.json','archives':'archives200.original.json'}
 def command(stage,records):
  calls.append(stage)
  if stage=='native':assert set(records)=={'observe'}
  if stage in ('hf','archives'):assert 'native' in records
  data={'scope':'SYNTHETIC_LOCAL_CPU_FIXTURE_NOT_ACTUAL200','stage':stage,'content_ok':stage!=bad_content}
  code='import pathlib,json,sys;pathlib.Path(sys.argv[1]).write_text(sys.argv[2]);print("fixture "+sys.argv[3]);raise SystemExit(int(sys.argv[4]))'
  return [sys.executable,'-B','-c',code,str(out/filenames[stage]),json.dumps(data),stage,'9' if stage==failed else '0']
 def accept(stage,p,child,refs):
  value=m.parse_json(m.stable_raw(p));assert value['stage']==stage
  m.typed(value['content_ok'],True,'Synthetic content rejected')
  return {'physical_path':str(p),'sha256':m.sha(m.stable_raw(p))}
 def finish(records):
  if final_fail:raise ValueError('Synthetic final-pin race rejected')
  for rec in records.values():m.stable_raw(rec['physical_path'],rec['sha256'])
 result=m.run_pipeline(out,command,accept,fixture_identity,finalize=finish)
 return out,result,calls


def test_actual_local_pipeline_success_scope_only(tmp_path):
 out,rc,calls=fixture_pipeline(tmp_path);assert rc==0 and calls==['observe','native','hf','archives']
 result=json.loads((out/'pipeline-result.original.json').read_text());assert result['status']=='FOUR_STORED_CONTENT_SCOPES_PASSED' and len(result['completed_records'])==4
 assert 'passed' not in result and 'receipt' not in result and result['scope']==m.SCOPE
 for stage in calls:
  wait=json.loads((out/(stage+'-exit.original.json')).read_text());assert wait['waited'] is True and wait['returncode']==0


def test_child_failure_stops_later_stages_and_retains_partial(tmp_path):
 out,rc,calls=fixture_pipeline(tmp_path,failed='native');assert rc==9 and calls==['observe','native']
 result=json.loads((out/'pipeline-result.original.json').read_text());assert set(result['completed_records'])=={'observe'} and result['status']=='FAILED_OR_UNKNOWN'
 assert (out/'native200.original.json').exists() and not (out/'hf-command.original.json').exists()
 with pytest.raises(ValueError):m.reserve_once(out)


def test_exit_zero_bad_content_stops_and_preserves_true_zero(tmp_path):
 out,rc,calls=fixture_pipeline(tmp_path,bad_content='hf');assert rc==1 and calls==['observe','native','hf']
 assert json.loads((out/'hf-exit.original.json').read_text())['returncode']==0
 assert json.loads((out/'pipeline-result.original.json').read_text())['status']=='FAILED_CONTENT'
 assert (out/'hf-content-failure.original.json').exists() and not (out/'archives-command.original.json').exists()


def test_final_report_recheck_failure_never_success(tmp_path):
 out,rc,calls=fixture_pipeline(tmp_path,final_fail=True);assert rc==1 and len(calls)==4
 assert json.loads((out/'pipeline-result.original.json').read_text())['status']=='FAILED_FINAL_RECORD_RECHECK'


def test_command_build_failure_retained(tmp_path):
 m.reserve_once(tmp_path/'once');out=tmp_path/'once'
 def reject(stage,refs):raise RuntimeError('Synthetic pinned tool disappeared')
 rc=m.run_pipeline(out,reject,lambda *x:None,fixture_identity)
 assert rc==1 and (out/'observe-control-failure.original.json').exists()
 assert not (out/'observe-exit.original.json').exists()


def fixture_report(stage):
 execution=fixture_identity(500);execution.update(execution_job_id='3304',execution_step_id='3',stored_training_job_id='3304',python=str(m.PYTHON))
 own=copy.deepcopy(execution);own['pid']=499
 value=dict(version=1,source_commit=m.SOURCE,config_sha256=m.CONFIG,job_id='3304',run_root=str(m.RUN),checkpoint_step=200,common_source_sha256=m.COMMON,execution_identity=execution,accepted5=dict(receipt_sha256=m.RECEIPT,public_key_sha256=m.PUBLIC))
 refs={'observe':{'sha256':'a'*64},'native':{'sha256':'b'*64}}
 if stage=='observe':
  rank=lambda r:dict(version=3,algorithm='pure_opsd',model_state_dict_type='SHARDED_STATE_DICT',rank=r,world_size=8,step=200,status='FULL_STATE_SERIALIZED_OK',scheduler_last_epoch=200,optimizer_state_entries=37,model_dtensor_entries=435,pid=1000+r,object_ids={'model':1,'optimizer':2,'scheduler':3,'teacher':4})
  value.update(kind=c.KIND,source_root=str(m.REPO),profile='multitraj200',horizon=200,warmup_steps=57,world_size=8,checkpoint_files={k:{} for k in c.checkpoint_paths()},archive_files={k:{} for k in c.archive_paths()},rank_reports=[rank(r) for r in range(8)],driver_report=dict(status='DRIVER_STATE_SERIALIZED_OK',step=200,job_id='3304',uid=os.getuid()),raw_statuses=dict(ranks=['FULL_STATE_SERIALIZED_OK']*8,driver='DRIVER_STATE_SERIALIZED_OK'),observer_source_sha256=m.PINS['observe_final200.py'])
 else:
  name,scope={'native':('inspect_final200_native.py',m.NATIVE_SCOPE),'hf':('load_final200_hf.py',m.HF_SCOPE),'archives':('check_final200_archives.py',m.ARCHIVE_SCOPE)}[stage]
  value.update(passed=True,cuda_is_initialized=False,scope=scope,inspector_source_sha256=m.PINS[name],inventory_sha256='a'*64)
  if stage=='native':value.update(clock=c.clock(),hf_all_rank_slices_exact=True,ranks=[{} for _ in range(8)])
  else:value.update(native_report_sha256='b'*64,native_inspector_source_sha256=m.PINS['inspect_final200_native.py'])
  if stage=='hf':value.update(all_loaded_parameters_finite=True,actual_cpu_forward_finite=True,tied_embeddings=True,hf_helper_sha256=m.HF_SHA)
  if stage=='archives':value.update(outer_count=200,member_count=4000,student_count=512000,teacher_count=102400,unique_common_question_count=102400,official_tokenizer_replayed=True,archived_questions_exact_stored200_sampler_and_common_parquet=True)
 return value,own,fixture_identity(500),refs


@pytest.mark.parametrize('stage',['observe','native','hf','archives'])
def test_exact_report_scopes_fixture_joins(stage):
 value,own,child,refs=fixture_report(stage);m.qualify(stage,value,c,own,child,refs)


@pytest.mark.parametrize('field,bad',[('passed',1),('cuda_is_initialized',0),('version',True),('checkpoint_step',200.0),('scope','EXACT_FULL_STATE_RELOAD_OK'),('inventory_sha256','f'*64),('inspector_source_sha256','f'*64)])
def test_exit_zero_does_not_replace_typed_native_content(field,bad):
 value,own,child,refs=fixture_report('native');value[field]=bad
 with pytest.raises(ValueError):m.qualify('native',value,c,own,child,refs)


@pytest.mark.parametrize('key,bad',[('pid',499),('start_ticks',True),('execution_job_id','901'),('stored_training_job_id','901'),('execution_step_id','30'),('cgroup','1:freezer:/job_3304/step_30\n'),('uids',[0,0,1,0]),('python','/wrong/python')])
def test_report_cannot_impersonate_controller_or_other_execution(key,bad):
 value,own,child,refs=fixture_report('hf');value['execution_identity'][key]=bad
 with pytest.raises(ValueError):m.qualify('hf',value,c,own,child,refs)


def test_inventory_cannot_claim_future_pass():
 value,own,child,refs=fixture_report('observe');value['passed']=True
 with pytest.raises(ValueError):m.qualify('observe',value,c,own,child,refs)


def test_all4000_archives_counts_must_exact_int():
 value,own,child,refs=fixture_report('archives');value['member_count']=4000.0
 with pytest.raises(ValueError):m.qualify('archives',value,c,own,child,refs)


def test_launch_typed_and_separate_execution_training_identity():
 value=dict(version=1,scope=m.SCOPE,stored_training_job_id='3304',execution_job_id='3304',source_commit=m.SOURCE,config_sha256=m.CONFIG,accepted5_receipt_sha256=m.RECEIPT,public_key_sha256=m.PUBLIC,controller_sha256='a'*64,bundle_manifest_sha256='b'*64,driver_pid=2548319,driver_start_ticks=1834784786,output=str(m.OUTPUT),precondition_sha256='c'*64)
 m.launch_fields(value)
 for key,bad in [('driver_start_ticks',True),('driver_pid',True),('version',True),('execution_job_id','901'),('stored_training_job_id','901')]:
  changed=dict(value);changed[key]=bad
  with pytest.raises(ValueError):m.launch_fields(changed)


@pytest.mark.parametrize('cgroup,valid',[
 ('1:freezer:/slurm_h100-10/uid_0/job_3304/step_3\n',True),
 ('0::/slurm/job-3304.scope/step-3.scope\n',True),
 ('1:freezer:/job_3304/step_4\n2:cpu:/job_901/step_3\n',False),
 ('1:freezer:/job_33040/step_3\n',False),
 ('1:freezer:/job_3304/step_30\n',False)])
def test_process_same_cgroup_path_boundaries(monkeypatch,cgroup,valid):
 tail=['S']+['0']*18+['123456']+['0']*5
 raw_stat='500 (fixture) '+' '.join(tail)
 def read(path,*args,**kwargs):
  return {'stat':raw_stat,'status':'State:\tS (sleeping)\nUid:\t0\t0\t0\t0\n','cgroup':cgroup}[path.name]
 monkeypatch.setattr(Path,'read_text',read);monkeypatch.setattr(m.os,'getuid',lambda:0)
 if valid:assert m.process(500,'3304','3')['start_ticks']==123456
 else:
  with pytest.raises(ValueError):m.process(500,'3304','3')


def test_production_host_guard_rejects_local_fixture_machine():
 if sys.platform!='linux' or m.socket.gethostname().split('.')[0]!='h100-10' or os.getuid()!=0:
  with pytest.raises(ValueError):m.host()


def prefix_fixture(tmp_path,monkeypatch):
 source=Path('work/opsd-unclipped-multitraj200-teacherrecompute/search_r1/baselines/checkpoint_gate.py').resolve()
 raw=source.read_bytes();assert m.sha(raw)==c.GATE
 gate=types.ModuleType('fixture_exact_stdlib_gate');gate.__file__=str(source);exec(compile(raw,str(source),'exec',dont_inherit=True),gate.__dict__)
 native=tmp_path/'synthetic-fsdp-source.py';native.write_bytes(b'# Explicit local clean-marker fixture; not a warning\n')
 checkpoint=tmp_path/'synthetic-checkpoint.py';checkpoint.write_bytes(b'# Explicit local native warning-path fixture\n')
 monkeypatch.setattr(m,'FSDP_WARNING_PATH',native);monkeypatch.setattr(m,'FSDP_WARNING_SHA',m.sha(native.read_bytes()));monkeypatch.setattr(m,'CHECKPOINT_WARNING_PATH',checkpoint)
 # Load the actual native211 schema unchanged: neither Torch warning module is
 # listed. These fixtures must join the actual shape, not an invented closure.
 actual_closure=json.loads((HERE.parent/'native-source-inventory-fcecbea5d939-ACTUAL.json').read_bytes())
 assert len(actual_closure['modules'])==211 and 'torch.utils.checkpoint' not in actual_closure['modules'] and 'torch.distributed.fsdp.fully_sharded_data_parallel' not in actual_closure['modules']
 context=dict(gate=gate,manifest={'driver_pid':2548319},workers=[{'pid':100+r} for r in range(8)],closure=actual_closure)
 values={'actor/optimizer_steps':10,'actor/skipped_empty_minibatches':0,'actor/samples':320,'actor/loss_mask_valid':1,'actor/opsd_loss_finite':1,'actor/grad_norm_finite':1,'teacher/trainable_parameters_per_rank':0,'teacher/gradient_tensors_per_rank':0,'teacher/student_optimizer_membership_per_rank':0,'teacher/storage_is_cpu_per_rank':1,'teacher/all_parameter_dtype_is_fp32_per_rank':1,'teacher/attention_is_eager_per_rank':1,'actor/allocator_is_native':1,'actor/allocator_configuration_is_default':1}
 metrics={key+'/rank_'+str(rank):value for rank in range(8) for key,value in values.items()}
 rows=[dict(actual_outer_updates=step,global_step=step,driver_pid=2548319,job_id='3304',source_commit=m.SOURCE,config_sha256=m.CONFIG,metrics=dict(metrics)) for step in range(1,201)]
 released=dict(driver_pid=2548319,actual_outer_updates=5,next_global_step=6,receipt_sha256=m.RECEIPT)
 def encoded():
  return b''.join(b'OPSD_E1_OUTER_JSON='+m.canonical(row) for row in rows[:5])+b'OPSD_E1_STEP5_RELEASED_JSON='+m.canonical(released)+b''.join(b'OPSD_E1_OUTER_JSON='+m.canonical(row) for row in rows[5:])
 return context,rows,released,encoded


def test_launch_prefix_real_parser_complete200_fixture(tmp_path,monkeypatch):
 context,rows,released,encoded=prefix_fixture(tmp_path,monkeypatch);value=m.validate_driver_prefix(encoded(),context,c)
 assert value['real_complete_outers']==200 and value['no201_in_read_prefix'] is True
 assert 'closed' not in value and 'passed' not in value


@pytest.mark.parametrize('mutation',['201','missing200','duplicate','wrongsource','wrongpid','rank9Adam','unsignedrelease','missingrelease'])
def test_launch_prefix_boundaries_reject(tmp_path,monkeypatch,mutation):
 context,rows,released,encoded=prefix_fixture(tmp_path,monkeypatch)
 if mutation=='201':rows.append(dict(rows[-1],global_step=201,actual_outer_updates=201))
 if mutation=='missing200':rows.pop()
 if mutation=='duplicate':rows[10]=dict(rows[9])
 if mutation=='wrongsource':rows[4]['source_commit']='0'*40
 if mutation=='wrongpid':rows[4]['driver_pid']=2548320
 if mutation=='rank9Adam':rows[4]['metrics']['actor/optimizer_steps/rank_7']=9
 if mutation=='unsignedrelease':released['receipt_sha256']='0'*64
 raw=encoded()
 if mutation=='missingrelease':raw=b'\n'.join(line for line in raw.split(b'\n') if b'OPSD_E1_STEP5_RELEASED_JSON=' not in line)+b'\n'
 with pytest.raises((ValueError,RuntimeError)):m.validate_driver_prefix(raw,context,c)
