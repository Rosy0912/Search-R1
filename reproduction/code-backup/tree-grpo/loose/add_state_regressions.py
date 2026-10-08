from pathlib import Path
p=Path('work/tree-instruct-300/tests/baselines/test_complete_training_state.py')
p.write_text(p.read_text()+'''

def test_corrupt_adam_is_refused_before_parameter_mutation():
    model=torch.nn.Linear(4,2);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-6)
    scheduler=get_constant_schedule_with_warmup(optimizer,57)
    model(torch.randn(3,4)).sum().backward();optimizer.step()
    saved=worker_snapshot(model,optimizer,scheduler,{'source':'fixed'})
    before=digest(worker_snapshot(model,optimizer,scheduler,{'source':'fixed'}))
    next(iter(saved['optimizer']['state'].values()))['exp_avg']=torch.ones(13)
    with pytest.raises(ValueError,match='Adam moment'):
        restore_worker(model,optimizer,scheduler,saved,{'source':'fixed'})
    assert digest(worker_snapshot(model,optimizer,scheduler,{'source':'fixed'}))==before


def test_incoherent_sampler_and_string_receipt_are_refused(tmp_path):
    sampler=CheckpointBatchSampler(40,4,42,'dataset')
    bad=sampler.state_dict();bad.update(cursor=20,epoch=91)
    with pytest.raises(ValueError,match='Incoherent'):sampler.load_state_dict(bad)
    ready=dict(job_id='1',code_commit='code',config_sha256='cfg',data_identity='data',model_identity='model')
    atomic_save(ready,tmp_path/'expected.json',json_format=True)
    approval=dict(ready,status='PASS',ready_sha256=file_sha(tmp_path/'expected.json'),
        full_state_reload_accepted='false',full_hf_reload_accepted='false',rank_tree_numerics_accepted='false')
    (tmp_path/'approved.json').write_text(json.dumps(approval))
    with pytest.raises(ValueError,match='acceptance missing'):wait_for_acceptance(tmp_path,ready,1)
    assert not (tmp_path/'released.json').exists()
''')
