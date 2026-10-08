import gzip,importlib.util,json,sys,copy
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parent))
import quality_final_audit_v1 as audit


def make_trace(tmp_path):
    folder=tmp_path/'iter_000030';folder.mkdir()
    files=[]
    for name in ['generated.jsonl.gz','scored.jsonl.gz','turn_01.jsonl.gz']:
        p=folder/name
        with gzip.open(p,'wt') as stream:
            for i in range(2560):
                row=dict(logical_step=30,trajectory_row=i,response_index=i%5,response_group=i//5,question_id=f'q{i//5}')
                if name=='generated.jsonl.gz':row.update(prompts=[1],responses=[2],attention_mask=[1,1],info_mask=[1,1])
                elif name=='scored.jsonl.gz':row.update({k:[1] for k in ('loss_mask','step_ids','policy_weights','advantages','token_level_rewards','returns','old_log_probs','ref_log_prob')})
                else:row.update(turn=1,active=True,done=False,valid_action=True,is_search=True,model_action='<search>query</search>',raw_tool_observation_before_token_limit='full original tool observation')
                stream.write(json.dumps(row)+'\n')
        files.append(dict(path=str(p),rows=2560,bytes=p.stat().st_size,sha256=audit.sha(p)))
    receipt=dict(logical_step=30,status='ACTUAL_OUTER_UPDATE_COMPLETE',expected_rows=2560,actual_optimizer_steps=10,
                 turn_count=1,files=files)
    (folder/'complete.json').write_text(json.dumps(receipt));return folder,receipt


def test_trace_physical_hash_and_count_fixture(tmp_path):
    folder,_=make_trace(tmp_path)
    assert audit.trace_receipt(folder,2560,'ACTUAL_OUTER_UPDATE_COMPLETE',10)['turn_count']==1


@pytest.mark.parametrize('case',['no_complete','wrong_optimizer','wrong_rows','missing_file','changed_file','partial','wrong_status'])
def test_rejects_incomplete_wrong_budget_or_changed_trace(tmp_path,case):
    folder,receipt=make_trace(tmp_path);p=folder/'complete.json'
    if case=='no_complete':p.unlink()
    elif case=='missing_file':(folder/'generated.jsonl.gz').unlink()
    elif case=='changed_file':(folder/'scored.jsonl.gz').write_bytes(b'changed')
    elif case=='partial':(folder/'bad.partial').write_text('failed write')
    else:
        if case=='wrong_optimizer':receipt['actual_optimizer_steps']=20
        if case=='wrong_rows':receipt['expected_rows']=128
        if case=='wrong_status':receipt['status']='SCORED_ONLY'
        p.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError):audit.trace_receipt(folder,2560,'ACTUAL_OUTER_UPDATE_COMPLETE',10)


def test_unique_dynamic_update_branch_and_no301():
    audit.validate_update_sequence([dict(outer_step=x) for x in range(26,301)],26,300)
    for values in [range(26,300),range(26,302),range(1,276),[26]+list(range(26,301))]:
        with pytest.raises(RuntimeError):audit.validate_update_sequence([dict(outer_step=x) for x in values],26,300)


@pytest.mark.parametrize('case',['duplicate_id','short','pairing','group','token_mask','turn','wrong_step','missing_action','missing_observation','short_scored_vectors'])
def test_internal_trace_inconsistency_rejected_even_with_matching_sha(tmp_path,case):
    folder,receipt=make_trace(tmp_path);p=folder/'scored.jsonl.gz'
    if case in ('token_mask','wrong_step'):p=folder/'generated.jsonl.gz'
    if case in ('turn','missing_action','missing_observation'):p=folder/'turn_01.jsonl.gz'
    with gzip.open(p,'rt') as stream:values=[json.loads(line) for line in stream]
    if case=='duplicate_id':values[-1]['trajectory_row']=0
    elif case=='short':values.pop()
    elif case=='pairing':values[0]['question_id']='another'
    elif case=='group':values[0]['response_index']=1
    elif case=='token_mask':values[0]['info_mask']=[]
    elif case=='turn':values[0]['turn']=2
    elif case=='wrong_step':values[0]['logical_step']=29
    elif case=='missing_action':values[0].pop('model_action')
    elif case=='missing_observation':values[0].pop('raw_tool_observation_before_token_limit')
    elif case=='short_scored_vectors':
        for k in ('loss_mask','step_ids','policy_weights','advantages','token_level_rewards','returns','old_log_probs','ref_log_prob'):values[0][k]=[]
    with gzip.open(p,'wt') as stream:
        for row in values:stream.write(json.dumps(row)+'\n')
    for item in receipt['files']:
        if item['path']==str(p):item.update(bytes=p.stat().st_size,sha256=audit.sha(p))
    (folder/'complete.json').write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError):audit.trace_receipt(folder,2560,'ACTUAL_OUTER_UPDATE_COMPLETE',10)


def evaluation_fixture():
    records=[dict(phase='stage64',logical_step=25)]
    records += [dict(phase='stage64',logical_step=s) for s in range(30,301,5)]
    records += [dict(phase='periodic512',logical_step=s) for s in range(50,301,50)]
    for i,r in enumerate(records):r['validation_trace_serial']=i
    return records


def test_phase_duplicates_and_reused_trace_serial_rejected():
    values=evaluation_fixture();audit.mandatory_evaluation_phases(values,25)
    for case in ('stage64','periodic512','serial'):
        bad=copy.deepcopy(values)
        if case=='serial':bad[-1]['validation_trace_serial']=0
        else:bad.append(copy.deepcopy(next(r for r in bad if r['phase']==case)))
        with pytest.raises(RuntimeError):audit.mandatory_evaluation_phases(bad,25)


def restored_fixture():
    saved=[dict(rank=r,pid=100+r,start_ticks=str(r),model='hash',adam='hash',rng='hash') for r in range(8)]
    values=[dict(r,pid=200+r['rank'],start_ticks=str(r['rank']+1)) for r in saved]
    return dict(workers=values),dict(files=[dict(pause_identity=r) for r in saved])


@pytest.mark.parametrize('case',['duplicate','missing','tensor','adam','rng'])
def test_restored_actual_tensor_identity_rejects_extra_rank_or_changed_hash(case):
    current,original=restored_fixture();audit.actual_restored_workers(current,original)
    if case=='duplicate':current['workers'].append(copy.deepcopy(current['workers'][0]))
    elif case=='missing':current['workers'].pop()
    else:current['workers'][0][{'tensor':'model','adam':'adam','rng':'rng'}[case]]='changed'
    with pytest.raises(RuntimeError):audit.actual_restored_workers(current,original)


def numeric_fixture(tmp_path):
    run=tmp_path/'run';p=run/'audit/numeric';p.mkdir(parents=True)
    workers={r:dict(pid=100+r) for r in range(8)}
    measure=dict(count=1,finite=1)
    stats={k:dict(measure) for k in ('current_log_probs','ratios','policy_loss','joint_action_ratios','joint_action_logratios')}
    stats.update(ratio_scope='nonzero_PG_actions_only; all_joint_logratios_recorded_separately',exact_zero_pg_actions=0,exact_zero_pg_tokens=0)
    for r in range(8):
        ident=dict(rank=r,pid=100+r,commit='new',config_sha256='config',data_identity=audit.CONTENT,run_id='run',outer_step=30)
        with (p/f'micro-{r}.jsonl').open('w') as stream:
            for i in range(160):stream.write(json.dumps(dict(ident,micro_index=i+1,optimizer_steps_completed=i//16,statistics=stats,loss_parts=dict(policy=measure,entropy=measure,total=measure,kl=measure),reference_minus_current=measure))+'\n')
        with (p/f'range-probe-{r}.jsonl').open('w') as stream:
            for i in range(10):stream.write(json.dumps(dict(ident,mini_index=i,optimizer_steps_completed=i,phase='accepted',local_bad=False,boundary_exit_invalid=False,local_bound='-inf',global_loss_scale=1.))+'\n')
    return run,workers


@pytest.mark.parametrize('case',['missing','rejected','scale','nonfinite','order','identity','empty_loss'])
def test_numeric_raw_stream_closure_negative_cases(tmp_path,case):
    run,workers=numeric_fixture(tmp_path);audit.audit_numeric_streams(run,30,30,'new','config',workers)
    name='micro-0.jsonl' if case in ('nonfinite','order','identity','empty_loss') else 'range-probe-0.jsonl'
    path=run/'audit/numeric'/name;values=audit.rows(path)
    if case=='missing':values.pop()
    elif case=='rejected':values[0]['phase']='rejected'
    elif case=='scale':values[0]['global_loss_scale']=.5
    elif case=='nonfinite':values[0]['loss_parts']['policy']['finite']=0
    elif case=='order':values[0]['micro_index']=2
    elif case=='identity':values[0]['pid']=999
    elif case=='empty_loss':
        for value in values[0]['loss_parts'].values():value.update(count=0,finite=0)
    path.write_text(''.join(json.dumps(r)+'\n' for r in values))
    with pytest.raises(RuntimeError):audit.audit_numeric_streams(run,30,30,'new','config',workers)

@pytest.mark.parametrize('case',['legacy_scope','wrong_source','other_checkpoint','twenty_adam','unreviewed'])
def test_native_scope_requires_actual_new_ten_adam_selected_state(tmp_path,case):
    result=dict(status='PASS',validation_scope='QUALITY10_GLOBAL256_SELECTED_FULL_STATE',code_commit='new',
        restore_step=25,restore_manifest_sha256='selected',expected_optimizer_steps_per_outer=10,
        expected_gradient_micro_batches_per_rank=160,global_row_denominator=256,production_denominator_per_mini=32,
        actual_outer_updates=0,cleanup_residual_states={},final_gpu=dict(processes=[]),job='new-native')
    if case=='legacy_scope':result['validation_scope']='LEGACY_NATIVE3229'
    elif case=='wrong_source':result['code_commit']='old229'
    elif case=='other_checkpoint':result['restore_manifest_sha256']='other'
    elif case=='twenty_adam':result['expected_optimizer_steps_per_outer']=20
    p=tmp_path/'result.json';p.write_text(json.dumps(result))
    review=dict(status='PASS',scope='QUALITY10_GLOBAL256_SELECTED_FULL_STATE',code_commit='new',
        restore_manifest_sha256='selected',result_path=str(p),result_sha256=audit.sha(p),job='new-native')
    if case=='unreviewed':review['status']='UNREVIEWED'
    r=tmp_path/'review.json';r.write_text(json.dumps(review))
    binding=dict(quality10_native_acceptance=dict(result_path=str(p),result_sha256=audit.sha(p),review_path=str(r),review_sha256=audit.sha(r)))
    with pytest.raises(RuntimeError):audit.verify_native_acceptance(binding,dict(step=25,manifest='selected'),'new')
