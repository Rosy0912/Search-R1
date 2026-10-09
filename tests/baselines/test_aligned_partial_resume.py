"""Small CPU admission tests; not an actual distributed/native reload witness."""
import ast,copy,hashlib,json
from pathlib import Path
import pytest
import torch
from search_r1.baselines import aligned_partial_resume as r
from search_r1.baselines.full_state import state_digest

class Mesh:
    device_type='cpu'
    mesh=torch.arange(8)
    mesh_dim_names=('dp',)
    def get_coordinate(self):return [0]
class Shard:
    """Tiny layout-bearing CPU fixture, not a claimed production DTensor."""
    device_mesh=Mesh()
    placements=('Shard(dim=0)',)
    shape=(32,)
    dtype=torch.float32
    def __init__(self,value=None):self.local=torch.ones(4)if value is None else value
    def to_local(self):return self.local
    def stride(self):return (1,)

def events():
    return [dict(rank=i,baseline_aligned=True,actual_outer=1,attempt_id=1,status='COMPLETE',optimizer_steps=20,
        expected_optimizer_steps=20,optimizer_opportunities_total=20,processed_opportunities=20,skipped_opportunities=0,
        unprocessed_opportunities=0,skipped_indices=[],skipped_reasons=[],postabort=False,update_complete=True,attempt_complete=True,
        outer_commit_phase='WRITES_PREPARED',accepted_complete_outer=False,scheduler_before=0,scheduler_after=0)for i in range(8)]

@pytest.fixture
def manifest(tmp_path):
    return dict(old_run_root=str(r.OLD_RUN),old_source_commit=r.OLD_SOURCE,target_run_root=str(tmp_path/'runs/formal-resume-9999'),
        target_job_id='9999',target_source_commit='b'*40,ranks=[dict(rank=i)for i in range(8)],rank_write_ledger=events())

@pytest.fixture
def native(manifest):
    pending=dict(actual_outer=1,attempt_id=1,scheduler_before=0,write_ledger_sha256=hashlib.sha256(json.dumps(manifest['rank_write_ledger'],sort_keys=True,allow_nan=False).encode()).hexdigest())
    payload=dict(version=3,rank=0,world_size=8,step=1,model_state_dict_type='SHARDED_STATE_DICT',update_status='PARTIAL',
        attempted_outer=1,previous_complete_outer=0,attempt_id=1,latest_rank_event=manifest['rank_write_ledger'][0],aligned_pending_outer=pending,
        scheduler=dict(last_epoch=0,_last_lr=[1e-6/57]),model={'weight':Shard()},optimizer=dict(state={0:dict(step=torch.tensor(20.),exp_avg=torch.ones(4),exp_avg_sq=torch.ones(4))},param_groups=[dict(lr=1e-6/57)]),
        rng={'cuda':[{'device':0,'state':torch.zeros(10,dtype=torch.uint8)}]},generation_rng=dict(version=1,torch_random_states=torch.zeros(10,dtype=torch.uint8),gen_random_states=torch.ones(10,dtype=torch.uint8)))
    report=dict(status='PARTIAL_CURRENT_STATE_SAVED',native_state_version=3,rank=0,world_size=8,step=1,roundtrip=False,pid=123,
        scheduler_last_epoch=0,model_dtensor_entries=1,optimizer_state_entries=1,latest_rank_event=payload['latest_rank_event'],aligned_pending_outer=pending,
        generation_rng_before_digest=state_digest(payload['generation_rng']),generation_rng_after_digest=state_digest(payload['generation_rng']))
    report['state_digest']=report['reloaded_state_digest']=state_digest(payload)
    return payload,report

def resign(payload,report):
    report['state_digest']=report['reloaded_state_digest']=state_digest(payload)

def test_manifest_and_native_pending_twenty_admitted(manifest,native):
    assert r.validate_manifest(manifest)==Path(manifest['target_run_root'])
    payload,report=native;assert r.validate_native(payload,report,0,manifest)is payload

@pytest.mark.parametrize('field,value',[('old_source_commit','a'*40),('old_run_root','/foreign'),('target_job_id','3356'),('target_job_id',9999),('target_source_commit','bad')])
def test_manifest_foreign_identity_refused(manifest,field,value):
    manifest[field]=value
    with pytest.raises(ValueError):r.validate_manifest(manifest)

@pytest.mark.parametrize('kind',['rank_missing','rank_duplicate','writes19','committed','attempt2'])
def test_manifest_incomplete_or_repeated_transaction_refused(manifest,kind):
    if kind=='rank_missing':manifest['ranks'].pop()
    elif kind=='rank_duplicate':manifest['ranks'][7]['rank']=0
    elif kind=='writes19':manifest['rank_write_ledger'][0]['optimizer_steps']=19
    elif kind=='committed':manifest['rank_write_ledger'][0]['outer_commit_phase']='STAGE_COMMITTED'
    else:manifest['rank_write_ledger'][0]['attempt_id']=2
    with pytest.raises(ValueError):r.validate_manifest(manifest)

@pytest.mark.parametrize('kind',['version2','outer25','counter_float','state_digest','pending_none','pending_wrong_hash','scheduler1','lr_zero','scheduler_lr_changed','adam19','adam21','moment_nan','model_inf','rng_changed','model_layout_missing'])
def test_native_rejects_inexact_state_even_with_resigned_payload(manifest,native,kind):
    payload,report=native
    if kind=='version2':payload['version']=2
    elif kind=='outer25':payload['step']=25
    elif kind=='counter_float':payload['attempt_id']=1.0
    elif kind=='state_digest':report['state_digest']='0'*64
    elif kind=='pending_none':payload['aligned_pending_outer']=None
    elif kind=='pending_wrong_hash':payload['aligned_pending_outer']['write_ledger_sha256']='f'*64
    elif kind=='scheduler1':payload['scheduler']['last_epoch']=1
    elif kind=='lr_zero':payload['optimizer']['param_groups'][0]['lr']=0.
    elif kind=='scheduler_lr_changed':payload['scheduler']['_last_lr']=[0.]
    elif kind=='adam19':payload['optimizer']['state'][0]['step']=torch.tensor(19.)
    elif kind=='adam21':payload['optimizer']['state'][0]['step']=torch.tensor(21.)
    elif kind=='moment_nan':payload['optimizer']['state'][0]['exp_avg'][0]=float('nan')
    elif kind=='model_inf':payload['model']['weight'].local[0]=float('inf')
    elif kind=='rng_changed':payload['generation_rng']['torch_random_states'][0]=1
    else:payload['model']={'weight':torch.ones(4)}
    if kind!='state_digest':resign(payload,report)
    with pytest.raises(ValueError):r.validate_native(payload,report,0,manifest)

def test_exact_file_hash_and_symlink_rejection(tmp_path):
    path=tmp_path/'rank.pt';raw=b'small original';path.write_bytes(raw)
    entry=dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    assert r.exact_file(entry,tmp_path)==path
    path.write_bytes(b'changed bytes!')
    with pytest.raises(ValueError):r.exact_file(entry,tmp_path)
    link=tmp_path/'symlink.pt';link.symlink_to(path);entry['path']=str(link)
    with pytest.raises(ValueError):r.exact_file(entry,tmp_path)

def test_fsdp_helper_signature_portability_and_no_training_write_calls():
    tree=ast.parse(Path(r.__file__).read_text());calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)]
    assert all(len(n.args)==1 for n in calls if isinstance(n.func,ast.Name)and n.func.id=='offload_fsdp_optimizer')
    assert all(len(n.args)==2 for n in calls if isinstance(n.func,ast.Name)and n.func.id=='load_fsdp_optimizer')
    assert not any(isinstance(n.func,ast.Attribute)and n.func.attr in ('step','observe','backward')for n in calls)
