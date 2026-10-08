"""Small local bytes/HF fixtures; no target official asset or full checkpoint scan."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import pytest

HERE=Path(__file__).parent;sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('phase2',HERE/'aligned_final_phase2.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


@pytest.fixture(autouse=True)
def cpu_only_env(monkeypatch):monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')


def test_full_small_inventory_fingerprints_and_membership(tmp_path,monkeypatch):
    root=tmp_path/'root';folder=root/'checkpoint';folder.mkdir(parents=True);(folder/'a').write_bytes(b'original')
    (folder/'ranks').mkdir();(folder/'ranks/rank0').write_bytes(b'CPU_FIXTURE_NOT_NATIVE')
    monkeypatch.setattr(h,'ROOT',root)
    manifest=h.inventory_manifest(folder,root);assert manifest['files_total']==2
    h.check_inventory(manifest)
    (folder/'a').write_bytes(b'original') # Same bytes/mtime-reset cannot retain ctime proof.
    with pytest.raises(ValueError,match='fingerprint'):h.check_inventory(manifest)


def test_full_inventory_refuses_symlink_and_new_membership(tmp_path,monkeypatch):
    root=tmp_path/'root';folder=root/'checkpoint';folder.mkdir(parents=True);(folder/'a').write_bytes(b'a')
    monkeypatch.setattr(h,'ROOT',root);record=h.inventory_manifest(folder,root)
    (folder/'b').write_bytes(b'b')
    with pytest.raises(ValueError,match='membership'):h.check_inventory(record)
    (folder/'b').unlink();(folder/'alias').symlink_to(folder/'a')
    with pytest.raises(ValueError,match='symlink'):h.inventory_manifest(folder,root)


def test_hardlink_asset_refused(tmp_path):
    path=tmp_path/'a';path.write_bytes(b'asset');os.link(path,tmp_path/'b')
    with pytest.raises(ValueError,match='single-link'):h.hash_file(path)


def tiny_model(path):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast,Qwen2Config,Qwen2ForCausalLM
    path.mkdir(parents=True);raw=Tokenizer(WordLevel({'[UNK]':0,'[PAD]':1,'[BOS]':2,'[EOS]':3,'The':4,'capital':5,'of':6,'France':7,'is':8},unk_token='[UNK]'));raw.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=raw,unk_token='[UNK]',pad_token='[PAD]',bos_token='[BOS]',eos_token='[EOS]')
    tokenizer.chat_template="{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}{% endfor %}"
    tokenizer.save_pretrained(path)
    torch.manual_seed(1);model=Qwen2ForCausalLM(Qwen2Config(vocab_size=16,hidden_size=8,intermediate_size=16,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,tie_word_embeddings=True))
    model.save_pretrained(path,safe_serialization=True)
    return model


def test_actual_tiny_official_CPU_load_tied_all_finite_and_forward(tmp_path):
    base=tmp_path/'base';tiny_model(base)
    report=h.official_cpu_hf(base,official=False)
    assert report['scope']=='CPU_FIXTURE_ONLY'and report['all_actual_parameters_finite']and report['finite_cpu_forward']
    assert report['tied_input_output_storage']and report['cost']['cpu_seconds']>=0
    with pytest.raises(ValueError,match='dimensions'):h.official_cpu_hf(base)


def submission(root):
    token='b'*32;job='9999'
    return dict(source=h.SOURCE,root=str(root),job=job,token=token,run=str(root/'runs'/f'formal-aligned-em200-{token}-{job}'),
        comment='BAPO_ALIGNED_EM200:formal:'+h.SOURCE+':'+token,scope='ACTUAL_SINGLE_FORMAL_JOB_SUBMITTED_NOT_TRAINING_COMPLETION')


def phase1(root,receipt):
    s=submission(root)
    return dict(schema='aligned-final-evidence-candidate-phase1-v1',source_commit=h.SOURCE,root=str(root),acceptance=False,goal_complete=False,release_receipt=False,
        status='TERMINAL_RAW_200_LEDGER_EM_FINAL_AGGREGATE_COLLECTED_PARTIAL_NOT_ACCEPTANCE',actual_submission=receipt,
        phase1=dict(source_commit=h.SOURCE,run_root=s['run'],job=s['job'],supported_scope='SINGLE_ORIGINAL_FORMAL_JOB_NO_INFRASTRUCTURE_FAULT_NO_RESUME_ONLY',
            accounting={'progress':{'actual_adam_writes_per_rank':[4000]*8,'completed_outers':200}}))


def test_immutable_phase1_source_actualjob_and_not_acceptance_binding(tmp_path,monkeypatch):
    monkeypatch.setattr(h,'ROOT',tmp_path);monkeypatch.setattr(h,'identity',lambda s:__import__('aligned_first5_cpu_evidence').identity(s,root=tmp_path))
    receipt={'path':str(tmp_path/'receipt'),'sha256':'a'*64,'bytes':12};report=phase1(tmp_path,receipt);submit=submission(tmp_path)
    assert h.phase1_identity(report,submit,receipt)['job']=='9999'
    for change in ({'source_commit':'old'},{'goal_complete':True},{'actual_submission':{**receipt,'sha256':'b'*64}}):
        with pytest.raises(ValueError):h.phase1_identity({**report,**change},submit,receipt)
    bad=copy.deepcopy(report);bad['phase1']['accounting']['progress']['actual_adam_writes_per_rank'][0]=3999
    with pytest.raises(ValueError):h.phase1_identity(bad,submit,receipt)


def test_HF_continuation_requires_full_hash_stage_and_all_five(tmp_path,monkeypatch):
    monkeypatch.setattr(h,'ROOT',tmp_path)
    bound={'binding':{'sha256':'a'*64},'actual_submission':{'sha256':'b'*64}}
    with pytest.raises(ValueError,match='hash-stage'):h.hf_stage({'status':'partial'},bound,{'run':tmp_path/'run'},tmp_path)
    report=dict(schema='aligned-final-evidence-candidate-phase2-v1',stage='hash',status='FULL_HASH_INVENTORIES_COLLECTED_CPU_HF_PENDING_NOT_ACCEPTANCE',source_commit=h.SOURCE,root=str(tmp_path),acceptance=False,goal_complete=False,
        phase1_binding=bound['binding'],actual_submission=bound['actual_submission'],hash_stage={'official_location':{'logical':'/base','resolved':'/base'},'checkpoints':[]})
    monkeypatch.setattr(h,'official_model_path',lambda p:Path('/base'))
    with pytest.raises(ValueError,match='five-checkpoint'):h.hf_stage(report,bound,{'run':tmp_path/'run'},tmp_path)


def five_inventory_fixture(tmp_path,monkeypatch):
    root=tmp_path/'root';run=root/'runs/formal-fixture';run.mkdir(parents=True);base=root/'official';tiny_model(base)
    # Small complete12-file manifest fixture, not the pinned official3B manifest.
    while len(list(base.iterdir()))<12:(base/f'fixture-{len(list(base.iterdir()))}.json').write_text('{}')
    official_files=[h.binding(p)for p in sorted(base.iterdir())]
    manifest=base/'VERIFIED-MODEL-MANIFEST.json';manifest.write_text(json.dumps({'files':[{'file':Path(r['path']).name,'sha256':r['sha256'],'bytes':r['bytes']}for r in official_files]}))
    monkeypatch.setattr(h,'ROOT',root);monkeypatch.setattr(h,'official_model_path',lambda p:base)
    def small_official(p):
        for record in official_files:assert h.binding(record['path'])==record
        return {'manifest':h.binding(manifest),'files':official_files}
    monkeypatch.setattr(h,'full_model_files',small_official)
    config={'actor_rollout_ref':{'model':{'path':str(base)}}}
    for outer in (5,50,100,150,200):
        hf=run/'checkpoints/actor'/f'global_step_{outer}';model=tiny_model(hf)
        import torch
        with torch.no_grad():model.model.layers[0].mlp.down_proj.weight.add_(outer*.001)
        model.save_pretrained(hf,safe_serialization=True)
        while len(list(hf.iterdir()))<12:(hf/f'fixture-{len(list(hf.iterdir()))}.json').write_text('{}')
        native=run/'checkpoints/fresh200_state'/f'outer_{outer}_COMPLETE';native.mkdir(parents=True);(native/'ranks').mkdir()
        for name in ('driver.pt','driver-reload.json','status.json','all-rank-write-ledger.json'):(native/name).write_bytes(b'CPU_METADATA_FIXTURE_NOT_STATE')
        for rank in range(8):
            (native/'ranks'/f'rank_{rank}.pt').write_bytes(b'CPU_FIXTURE_NOT_NATIVE')
            (native/'ranks'/f'rank_{rank}.json').write_text('{}')
        algorithm=run/'checkpoints/algorithm'/f'global_step_{outer}';algorithm.mkdir(parents=True);(algorithm/'controller.pt').write_bytes(b'CPU_FIXTURE_NOT_ALGORITHM')
    phase_path=root/'evidence/phase1';phase_path.mkdir(parents=True);report_path=phase_path/'phase1-evidence.json';report_path.write_text('CPU_FIXTURE_ONLY')
    receipt=root/'receipt';receipt.write_text('CPU_FIXTURE_ONLY')
    bound={'report':{'phase1':{'config':config}},'original_bindings':[],'binding':h.binding(report_path),'actual_submission':h.binding(receipt)}
    who={'run':run,'job':'9999'};output=root/'evidence/hash';output.mkdir()
    stage=h.full_hash_stage(bound,who,output)
    hash_report=dict(schema='aligned-final-evidence-candidate-phase2-v1',stage='hash',status=stage['status'],source_commit=h.SOURCE,root=str(root),
        acceptance=False,goal_complete=False,phase1_binding=bound['binding'],actual_submission=bound['actual_submission'],hash_stage=stage)
    return root,base,bound,who,hash_report


def test_five_all_group_hash_inventories_actual_small_bytes(tmp_path,monkeypatch):
    root,base,bound,who,report=five_inventory_fixture(tmp_path,monkeypatch)
    assert len(report['hash_stage']['official_records'])==13
    assert [cp['actual_outer']for cp in report['hash_stage']['checkpoints']]==[5,50,100,150,200]
    for cp in report['hash_stage']['checkpoints']:
        assert cp['groups']['hf']['files_total']==12 and cp['groups']['native_driver']['files_total']==20
        for group in cp['groups'].values():h.check_inventory(group)
    path=Path(report['hash_stage']['checkpoints'][-1]['groups']['native_driver']['files'][0]['path'])
    path.write_bytes(b'changed original native bytes')
    with pytest.raises(ValueError,match='fingerprint'):h.check_inventory(report['hash_stage']['checkpoints'][-1]['groups']['native_driver'])


def test_actual_tiny_five_HF_continuation_full_tensor_scan_and_finite_load(tmp_path,monkeypatch):
    root,base,bound,who,report=five_inventory_fixture(tmp_path,monkeypatch)
    official=h.official_cpu_hf;export=h.cpu_hf
    monkeypatch.setattr(h,'official_cpu_hf',lambda p:official(p,official=False))
    monkeypatch.setattr(h,'cpu_hf',lambda b,p:export(b,p,official=False))
    output=root/'evidence/hf';output.mkdir()
    result=h.hf_stage(report,bound,who,output)
    assert result['official']['scope']=='CPU_FIXTURE_ONLY'and len(result['results'])==5
    assert all(r['scope']=='CPU_FIXTURE_ONLY'and r['all_parameters_finite']and r['forward']['finite']and r['parameter_delta']['changed_elements']>0 for r in result['results'])
    assert len(list(output.glob('outer-*-actual-cpu-hf.json')))==5
    assert 'NOT_CPU_DESERIALIZED' in result['native_role']
    # This is real tinyHF math/load work using synthetic manifests, never3B/native acceptance.


def test_pending_terminal_refuses_assets_before_continuation_reads(tmp_path,monkeypatch):
    root=tmp_path/'root';(root/'scheduler-head').mkdir(parents=True);parent=root/'evidence/phase1';parent.mkdir(parents=True)
    submit=submission(root);receipt=root/'scheduler-head/formal-submit-receipt.json';receipt.write_text(json.dumps(submit));receipt_binding=h.binding(receipt)
    report=phase1(root,receipt_binding);report['phase1']['original_bindings']=[{'path':str(root/'LARGE_ASSET_MUST_NOT_BE_READ'),'bytes':1,'sha256':'a'*64}]
    report['helper_files']=[];report['formal_tools']=[]
    path=parent/'phase1-evidence.json';path.write_text(json.dumps(report));output=root/'evidence/hf';output.mkdir()
    monkeypatch.setattr(h,'ROOT',root);monkeypatch.setattr(h,'identity',lambda s:__import__('aligned_first5_cpu_evidence').identity(s,root=root))
    original_terminal=h.terminal_observation
    monkeypatch.setattr(h,'terminal_observation',lambda s,q:original_terminal(s,q,root=root))
    import collect_hardware_evidence
    row=dict(JobId='9999',JobName='bapo-aligned-em200-formal',Comment=submit['comment'],WorkDir=str(root),ReqNodeList='h100-5',NodeList='(null)',
        JobState='PENDING',ExitCode='0:0',SubmitTime='2026-10-09T01:00:00',StartTime='Unknown',EndTime='Unknown',Command=str(root/'tools/deployment/formal.sbatch')+' '+submit['token'])
    monkeypatch.setattr(collect_hardware_evidence,'execute',lambda argv:{'exit_code':0,'stdout':' '.join(k+'='+v for k,v in row.items()),'stderr':''})
    with pytest.raises(ValueError,match='terminal'):h.bind_phase1(path,output)
    assert (output/'scheduler-original-query.json').exists()and (output/'original-phase1-report.json').exists()
    assert not (root/'LARGE_ASSET_MUST_NOT_BE_READ').exists()


def test_pending_stage_boundaries_and_incomplete_export_completion():
    before=h.pending_tasks({'hash_complete':False,'official_hf':False,'export_hf':[]})
    assert all(task in before for task in h.HASH_TASKS+h.CPU_TASKS+h.PENDING)
    hash_only=h.pending_tasks({'hash_complete':True,'official_hf':False,'export_hf':[]})
    assert not set(h.HASH_TASKS)&set(hash_only)and set(h.CPU_TASKS)<=set(hash_only)
    partial=h.pending_tasks({'hash_complete':True,'official_hf':True,'export_hf':[5,50]})
    assert h.CPU_TASKS[0]not in partial and h.CPU_TASKS[1]not in partial and h.CPU_TASKS[2]not in partial
    assert all(task in partial for task in h.CPU_TASKS[3:]+h.PENDING)
    done=h.pending_tasks({'hash_complete':True,'official_hf':True,'export_hf':[5,50,100,150,200]})
    assert done==h.PENDING # Native/recovery/cleanup/review never disappear.
    invalidated=h.pending_tasks({'hash_complete':False,'official_hf':True,'export_hf':[5,50]})
    assert set(h.HASH_TASKS+h.CPU_TASKS)<=set(invalidated)


def test_failed_HF_retains_uncompleted_exports_in_machine_pending(tmp_path,monkeypatch):
    root,base,bound,who,report=five_inventory_fixture(tmp_path,monkeypatch)
    official=h.official_cpu_hf;export=h.cpu_hf
    monkeypatch.setattr(h,'official_cpu_hf',lambda p:official(p,official=False))
    def load(b,p):
        if p.name=='global_step_100':raise RuntimeError('CPU fixture failed100')
        return export(b,p,official=False)
    monkeypatch.setattr(h,'cpu_hf',load);output=root/'evidence/hf';output.mkdir();progress=dict(hash_complete=False,official_hf=False,export_hf=[])
    with pytest.raises(RuntimeError,match='failed100'):h.hf_stage(report,bound,who,output,progress)
    assert progress==dict(hash_complete=True,official_hf=True,export_hf=[5,50])
    pending=h.pending_tasks(progress)
    assert all(task in pending for task in h.CPU_TASKS[3:])and not set(h.HASH_TASKS)&set(pending)
    assert (output/'outer-50-actual-cpu-hf.json').exists()and not (output/'outer-100-actual-cpu-hf.json').exists()


def test_failed_hash_retains_hash_and_all_cpu_pending(tmp_path,monkeypatch):
    root,base,bound,who,report=five_inventory_fixture(tmp_path,monkeypatch)
    # A fresh partial inventory output, without a reusable successful hash result.
    output=root/'evidence/hash-failure';output.mkdir();progress=dict(hash_complete=False,official_hf=False,export_hf=[])
    (who['run']/'checkpoints/algorithm/global_step_100/controller.pt').unlink()
    with pytest.raises(ValueError,match='nonempty'):h.full_hash_stage(bound,who,output,progress)
    assert set(h.HASH_TASKS+h.CPU_TASKS)<=set(h.pending_tasks(progress))
    assert (output/'outer-50-algorithm-full-sha256.json').exists()


@pytest.mark.parametrize('changed',['hash_report','hash_helper','logical_model','archive_copy'])
def test_exception_closure_revokes_progress_for_every_continuation_original(tmp_path,monkeypatch,changed):
    root=tmp_path/'root';root.mkdir();monkeypatch.setattr(h,'ROOT',root)
    names=('phase1','receipt','raw_original','archive_copy','current_helper','hash_report','hash_helper')
    paths={name:root/name for name in names}
    for p in paths.values():p.write_bytes(b'original')
    records={name:h.binding(p)for name,p in paths.items()}
    bound=dict(original_bindings=[records['raw_original']],archive_bindings=[records['archive_copy']],binding=records['phase1'],actual_submission=records['receipt'])
    base=root/'model';base.mkdir();weight=base/'weight';weight.write_bytes(b'tiny fixture')
    logical=root/'logical';logical.symlink_to(base,target_is_directory=True)
    monkeypatch.setattr(h,'official_model_path',lambda p:p.resolve(strict=True))
    data=dict(official_location={'logical':str(logical),'resolved':str(base)},official_records=[h.hash_file(weight)],checkpoints=[])
    progress=dict(hash_complete=True,official_hf=True,export_hf=[5,50,100,150,200])
    helpers=[records['current_helper']];prior=[records['hash_report'],records['hash_helper']]
    h.completion_inputs(bound,helpers,prior,data)
    if changed=='logical_model':
        other=root/'other';other.mkdir();logical.unlink();logical.symlink_to(other,target_is_directory=True)
    else:paths[changed].write_bytes(b'changed during long CPU work')
    result,reason=h.invalidate_changed_completion(progress,bound,helpers,prior,data)
    assert result==dict(hash_complete=False,official_hf=False,export_hf=[])
    assert reason is not None and set(h.HASH_TASKS+h.CPU_TASKS)<=set(h.pending_tasks(result))
    # Original proof fields are retained; mutation never silently updates a digest.
    assert records['hash_report']['sha256']==h.binding(paths['phase1'])['sha256']
