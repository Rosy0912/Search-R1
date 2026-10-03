"""Real DataProto + explicitly simulated CPU workers. No real Slurm/GPU proof."""
import copy,os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest,torch
from verl import DataProto
from search_r1.baselines import corrected_runtime_capture as capture
from search_r1.baselines import corrected_evaluation as ce
SIMULATED_JOB='101' # CPU fixture, never submitted or reported as an actual/future job.

@pytest.fixture
def spec():
 repo=Path(capture.__file__).resolve().parents[2]
 names=set(module.replace('.','/')+'.py' for module in capture.WORKER_MODULES)
 return dict(eval_repo=str(repo),source_files={name:ce.digest(repo/name) for name in names},job_id=SIMULATED_JOB,stage_sha256='a'*64,uid=os.getuid())

def process(pid,job=SIMULATED_JOB):
 fields=['S','1']+['0']*18;fields[19]='123';raw_stat=f'{pid} (python) '+' '.join(fields);cmd=['python','actual_worker.py'];uid=os.getuid();cg=f'0::/slurm/job_{job}/step_batch\n'
 return dict(pid=pid,ppid=1,process_start_ticks='123',uid=uid,uids=[uid]*4,cgroup=cg,cmdline=cmd,raw_stat=raw_stat,raw_status='Uid:\t'+'\t'.join([str(uid)]*4)+'\n',raw_cgroup=cg,raw_cmdline_hex=(b'\0'.join(x.encode() for x in cmd)+b'\0').hex())

def records(spec):
 repo=Path(spec['eval_repo']);imports={module:dict(path=str(repo/(module.replace('.','/')+'.py')),sha256=spec['source_files'][module.replace('.','/')+'.py']) for module in capture.WORKER_MODULES}
 return [dict(rank=rank,world_size=8,pid=200+rank,uid=os.getuid(),node='h100-10',job_id=SIMULATED_JOB,stage_sha256=spec['stage_sha256'],source_imports=copy.deepcopy(imports),process=process(200+rank),worker_type='verl.workers.fsdp_workers.ActorRolloutRefWorker',rollout_type='verl.workers.rollout.vllm_rollout.vllm_rollout.vLLMRollout',device_index=0,cuda_visible_devices=str(rank),gpu_uuid=f'GPU-CPU-fixture-{rank}',callback_sha256=ce.digest(capture.__file__)) for rank in range(8)]

def test_cpu_eight_rank_schema_only(spec):capture.validate_worker_rows(records(spec),spec)

@pytest.mark.parametrize('mutation',['missing','duplicate','world','node','job','stage','uid','pid','start','rawstat','rawuid','rawcmd','cgroup','class','rollout','uuid','callback','import','foreignimport'])
def test_worker_identity_source_ownership_rejections(spec,mutation):
 rows=records(spec);r=rows[0]
 if mutation=='missing':rows.pop()
 if mutation=='duplicate':r['rank']=1
 if mutation=='world':r['world_size']=7
 if mutation=='node':r['node']='other-node'
 if mutation=='job':r['job_id']='other'
 if mutation=='stage':r['stage_sha256']='0'*64
 if mutation=='uid':r['uid']+=1
 if mutation=='pid':r['pid']=201
 if mutation=='start':r['process']['process_start_ticks']='124'
 if mutation=='rawstat':r['process']['raw_stat']='fake raw stat'
 if mutation=='rawuid':r['process']['raw_status']='Uid: 999 999 999 999'
 if mutation=='rawcmd':r['process']['raw_cmdline_hex']='00'
 if mutation=='cgroup':r['process']['cgroup']='0::/slurm/job_other/step_batch'
 if mutation=='class':r['worker_type']='fake.Worker'
 if mutation=='rollout':r['rollout_type']='fake.Rollout'
 if mutation=='uuid':r['gpu_uuid']=rows[1]['gpu_uuid']
 if mutation=='callback':r['callback_sha256']='0'*64
 if mutation=='import':r['source_imports'].pop(capture.WORKER_MODULES[0])
 if mutation=='foreignimport':r['source_imports'][capture.WORKER_MODULES[0]]['path']='/tmp/fake.py'
 with pytest.raises((ValueError,IndexError)):capture.validate_worker_rows(rows,spec)

def test_real_dataproto_fake_dispatch_chunks_and_collects(spec,monkeypatch):
 class FakeGroup:
  world_size=8
  def execute_with_func_generator(self,func,probe):
   assert func is capture.worker_import_callback
   parts=probe.chunk(8);self.seen=[len(p) for p in parts]
   rows=records(spec)
   return DataProto.concat([DataProto.from_dict(tensors={'audit_probe':part.batch['audit_probe'].clone()},non_tensors={'worker_import_audit':np.array([rows[i]],dtype=object)}) for i,part in enumerate(parts)])
 trainer=SimpleNamespace(actor_rollout_wg=FakeGroup());monkeypatch.setenv('SLURM_JOB_ID',SIMULATED_JOB)
 result=capture.capture_workers(trainer,dict(source_files=spec['source_files'],reviewed_stage_sha256=spec['stage_sha256']))
 assert trainer.actor_rollout_wg.seen==[1]*8 and len(result['rows'])==8

def test_capture_entry_rejects_missing_actual_job(spec,monkeypatch):
 monkeypatch.delenv('SLURM_JOB_ID',raising=False)
 with pytest.raises(ValueError,match='own Slurm'):capture.capture_workers(SimpleNamespace(),{})

class Group:
 def generate_sequences(self,probe):return probe

def test_actual_greedy_generation_guard_readonly(spec):
 group=Group();audit=dict(spec=spec,rows=records(spec));guard=capture.GenerationCallGuard(group,audit)
 probe=DataProto.from_dict(tensors={'input_ids':torch.ones((8,1),dtype=torch.long)},meta_info={'do_sample':False,'validate':True})
 assert group.generate_sequences(probe) is probe and torch.equal(probe.batch['input_ids'],torch.ones((8,1),dtype=torch.long))
 record=guard.record();capture.validate_completed_runtime(record,dict(source_files=spec['source_files'],reviewed_stage_sha256=spec['stage_sha256']))
 assert record['generation_calls']==[dict(sequence=1,rows=8,do_sample=False,validate=True)]

@pytest.mark.parametrize('meta',[{}, {'do_sample':True,'validate':True},{'do_sample':False,'validate':False},{'do_sample':0,'validate':1}])
def test_generate_guard_rejects_actual_sampling_or_missing_flag(meta,spec):
 group=Group();guard=capture.GenerationCallGuard(group,dict(spec=spec,rows=records(spec)))
 probe=DataProto.from_dict(tensors={'input_ids':torch.ones((8,1),dtype=torch.long)},meta_info=meta)
 with pytest.raises(RuntimeError):group.generate_sequences(probe)
 with pytest.raises(ValueError):guard.record()

def test_generation_guard_replaced_or_unused_rejected(spec):
 group=Group();guard=capture.GenerationCallGuard(group,dict(spec=spec,rows=records(spec)))
 with pytest.raises(ValueError):guard.record()
 group.generate_sequences=lambda probe:probe
 with pytest.raises(ValueError):guard.record()

def test_stage_path_arguments_are_explicit():
 source=Path(ce.__file__).read_text();assert "'--stage',str(stage),'--stage-sha256',provenance['reviewed_stage_sha256']" in source

def test_actual_callback_real_dataproto_cpu_mocked_device_only(spec,monkeypatch):
 import sys
 from types import SimpleNamespace
 for name in capture.WORKER_MODULES:
  monkeypatch.setitem(sys.modules,name,SimpleNamespace(__file__=str(Path(spec['eval_repo'])/(name.replace('.','/')+'.py'))))
 monkeypatch.setenv('SLURM_JOB_ID',SIMULATED_JOB);monkeypatch.setenv('OPSD_CORRECTED_EVAL_STAGE_SHA256',spec['stage_sha256']);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
 monkeypatch.setattr(capture,'process_identity',lambda pid:process(pid));monkeypatch.setattr(capture.socket,'gethostname',lambda:'h100-10')
 monkeypatch.setattr(torch.cuda,'current_device',lambda:0);monkeypatch.setattr(torch.cuda,'get_device_properties',lambda device:SimpleNamespace(uuid='GPU-CPU-fixture-0'))
 Rollout=type('vLLMRollout',(),{'__module__':'verl.workers.rollout.vllm_rollout.vllm_rollout'})
 Worker=type('ActorRolloutRefWorker',(),{'__module__':'verl.workers.fsdp_workers','rank':0,'world_size':8,'rollout':Rollout()})
 probe=DataProto.from_dict(tensors={'audit_probe':torch.zeros((1,1),dtype=torch.int64)},meta_info={'corrected_audit_spec':spec})
 output=capture.worker_import_callback(Worker(),probe)
 assert isinstance(output,DataProto) and len(output)==1 and output.non_tensor_batch['worker_import_audit'][0]['gpu_uuid']=='GPU-CPU-fixture-0'

@pytest.mark.parametrize('rows',[1,7,65,72])
def test_generate_padded_batch_budget_rejected(spec,rows):
 group=Group();capture.GenerationCallGuard(group,dict(spec=spec,rows=records(spec)))
 probe=DataProto.from_dict(tensors={'input_ids':torch.ones((rows,1),dtype=torch.long)},meta_info={'do_sample':False,'validate':True})
 with pytest.raises(RuntimeError,match='batch budget'):group.generate_sequences(probe)
