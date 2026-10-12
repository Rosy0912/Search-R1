"""Small node11 binding/cohost prefix tests; no official model or live process loads."""
import importlib.util
import json
from pathlib import Path
import pytest

@pytest.fixture
def cpu():
    p=Path(__file__).with_name('aligned_first5_cpu_evidence.py');s=importlib.util.spec_from_file_location('node11_first5_fixture',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def submission(c):
    return dict(source=c.SOURCE,root=str(c.ROOT),scope='ACTUAL_SINGLE_FORMAL_JOB_SUBMITTED_NOT_TRAINING_COMPLETION',job=c.JOB,token=c.TOKEN,host='h100-11',deployment_directory=str(c.FORMAL_DEPLOY),files={n:'a'*64 for n in c.FORMAL_TOOL_NAMES},run=str(c.ROOT/'runs'/('formal-aligned-em200-'+c.TOKEN+'-'+c.JOB)),comment='BAPO_ALIGNED_EM200:formal:'+c.SOURCE+':'+c.TOKEN)

def test_unique_actual3359_and_canonical_watch_mapping(cpu):
    assert cpu.identity(submission(cpu))['job']=='3359'
    assert cpu.formal_tool_path('tools/resource_watch.py')==cpu.ROOT/'tools/resource_watch.py'
    assert cpu.formal_tool_path('formal_driver.py')==cpu.FORMAL_DEPLOY/'formal_driver.py'
    with pytest.raises(ValueError):cpu.formal_tool_path('../formal_driver.py')

@pytest.mark.parametrize('change',[{'job':'3348'},{'host':'h100-5'},{'token':'0'*32},{'deployment_directory':'/old/deployment'}])
def test_old_job_host_token_or_tool_namespace_refused(cpu,change):
    row=submission(cpu);row.update(change)
    with pytest.raises(ValueError):cpu.identity(row)

@pytest.fixture
def resources(tmp_path,cpu):
    run=tmp_path/'run';run.mkdir();out=tmp_path/'out';out.mkdir();uuids=['GPU-fixture-'+str(i)for i in range(8)]
    boot=dict(run=str(run),job='3359',policy='COHOST_NODE11_INITIAL_FREE_NOT_IDLE',leases=[dict(uuid=u,fd=20+i)for i,u in enumerate(uuids)])
    (run/'cohost-lease-bootstrap.json').write_text(json.dumps(boot))
    row=dict(run=str(run),slurm_job_id='3359',policy='COHOST_FREE_RESERVE_NO_FOREIGN_SM_OR_MEMORY_PERCENT_GATE',pause=False,reason='within_free_reserve',reserve_free_mib=1024,gpus=[dict(uuid=u,free_mib=39236)for u in uuids],processes=[dict(own=False,memory_mib=40000,sm_percent=99)])
    return cpu,run,out,row

def test_foreign_high_occupancy_not_idle_gated_and_prefix_append_allowed(resources):
    c,run,out,row=resources;p=run/'resource-watch.jsonl';p.write_bytes((json.dumps(row)+'\n').encode()+b'{"partial":')
    proof=c.cohost_watch_prefix(dict(run=run,job='3359'),out)
    assert proof['incomplete_tail_bytes'] and proof['rows'][0]['processes'][0]['own']is False
    p.write_bytes(p.read_bytes()+b'1}\n');c.verify_watch_prefix(proof)
    raw=p.read_bytes();p.write_bytes(b'!'+raw[1:])
    with pytest.raises(ValueError,match='prefix changed'):c.verify_watch_prefix(proof)

@pytest.mark.parametrize('change',[{'pause':True},{'slurm_job_id':'3345'},{'policy':'OLD15PERCENT_IDLE'},{'error':'unknown query'},{'reserve_free_mib':0}])
def test_cohost_fault_unknown_or_foreign_prefix_refused(resources,change):
    c,run,out,row=resources;row.update(change);(run/'resource-watch.jsonl').write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError):c.cohost_watch_prefix(dict(run=run,job='3359'),out)


def test_outside_file_requires_source_validated_exact_bytes(cpu,tmp_path):
    root=tmp_path/'new';root.mkdir();old=tmp_path/'old';old.mkdir()
    original=old/'driver.pt';original.write_bytes(b'tiny original fixture')
    with pytest.raises(ValueError,match='source-validated'):
        cpu.regular_member(original,root,{})
    closure={str(original):cpu.digest(original)}
    assert cpu.regular_member(original,root,closure)==original
    original.write_bytes(b'changed')
    with pytest.raises(ValueError,match='source-validated'):
        cpu.regular_member(original,root,closure)


def test_closure_is_returned_only_by_actual_gate_interface(cpu,tmp_path):
    from types import SimpleNamespace
    seen=[];originals={'/old/explicit.pt':'a'*64}
    manifest={'files':[{'path':'/old/explicit.pt','sha256':'a'*64}],
              'rank_roundtrips':[{'rank':0,'pid':123}],
              'original_closure':{'/foreign/declared-but-unverified.pt':'b'*64}}
    def check(run,source,members,reports,pid):
        seen.append((run,source,members,reports,pid));return {'originals':originals}
    result=cpu.validated_original_closure(manifest,SimpleNamespace(complete5_lineage=check),tmp_path,456)
    assert result is originals and '/foreign/declared-but-unverified.pt'not in result
    assert seen==[(tmp_path,cpu.SOURCE,{'/old/explicit.pt':'a'*64},manifest['rank_roundtrips'],456)]


@pytest.mark.parametrize('failure',['absent','rejected','duplicate'])
def test_unvalidated_lineage_never_opens_cross_root(cpu,tmp_path,failure):
    from types import SimpleNamespace
    files=[{'path':'/old/explicit.pt','sha256':'a'*64}]
    if failure=='duplicate':files*=2
    def check(*args):
        if failure=='rejected':raise ValueError('source lineage rejected')
        return None
    with pytest.raises(ValueError):
        cpu.validated_original_closure({'files':files,'rank_roundtrips':[]},SimpleNamespace(complete5_lineage=check),tmp_path)


def test_complete5_timeline_preserves_original_identities(cpu,tmp_path):
    from types import SimpleNamespace
    old=tmp_path/'old3357';old.mkdir();run=tmp_path/'new3359';run.mkdir()
    attempts=[dict(outer=i,source_commit='original3357',pid=111)for i in range(1,6)]
    events=[dict(step=i,pid=111)for i in range(1,6)]
    log=old/'train.log';log.write_text(''.join('FRESH200_ATTEMPT_JSON='+json.dumps(a)+'\nBAPO_TRAIN_METRICS_JSON='+json.dumps(e)+'\n'for a,e in zip(attempts,events)))
    sha=cpu.digest(log);manifest=dict(files=[dict(path=str(log),sha256=sha)],rank_roundtrips=[])
    lineage=dict(old_run=old,old_driver_pid=111,old_source='original3357',originals={str(log):sha})
    gate=SimpleNamespace(complete5_lineage=lambda *args:lineage)
    actual,aa,ee=cpu.complete5_timeline(manifest,gate,run,222)
    assert actual is lineage and aa==attempts and ee==events
    log.write_text(log.read_text()+'changed\n')
    with pytest.raises(ValueError,match='timeline missing/unbound'):cpu.complete5_timeline(manifest,gate,run,222)


def test_partial_only_gate_cannot_authorize_complete5(cpu,tmp_path):
    from types import SimpleNamespace
    with pytest.raises(AttributeError):
        cpu.validated_original_closure(dict(files=[],rank_roundtrips=[]),SimpleNamespace(partial_lineage=lambda *args:dict(originals={})),tmp_path)


@pytest.mark.parametrize('name',['FRESH200_ATTEMPT_JSON=','BAPO_TRAIN_METRICS_JSON='])
def test_recovery_live_log_must_not_replay_training(cpu,name):
    cpu.verify_only_summaries('FRESH200_STEP5_PAUSED_JSON={}\n')
    with pytest.raises(ValueError,match='repeat'):cpu.verify_only_summaries(name+'{}\n')
