"""CPU evidence fixtures only; never real Slurm/GPU acceptance or collection."""
import importlib.util
import json
from pathlib import Path
import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('collector', HERE / 'collect_hardware_evidence.py')
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def query(state='COMPLETED', comment=None, control_exit=0, end='2026-10-09T04:00:00', code='0:0'):
    def run(argv):
        identity = comment if comment is not None else collector.COMMENT
        stdout = (f'JobId=3345 JobName=bapo-aligned-em200-hardware JobState={state} Comment={identity} WorkDir={collector.ROOT} Command={collector.ROOT}/tools/deployment/hardware.sbatch {collector.TOKEN} ReqNodeList=h100-5 NodeList=h100-5 ExitCode={code} StartTime=2026-10-09T03:50:00 EndTime={end} SubmitTime=2026-10-09T03:20:00\n' if argv[0] == 'scontrol' else
                  f'3345|bapo-aligned-em200-hardware|{state}|{code}|h100-5|{identity}|2026-10-09T03:50:00|{end}|2026-10-09T03:20:00|\n')
        return dict(command=argv, exit_code=control_exit if argv[0] == 'scontrol' else 0, stdout=stdout, stderr='')
    return run


@pytest.mark.parametrize('state', ['PENDING', 'RUNNING', 'COMPLETING', 'SUSPENDED'])
def test_nonterminal_is_never_acceptance_or_terminal_archive(tmp_path, state):
    value = collector.observe(query(state))
    assert value['acceptance'] is False and value['terminal_verified'] is False
    with pytest.raises(ValueError, match='nonterminal'):
        collector.archive_terminal(tmp_path, tmp_path / 'evidence/new', value, query(state))
    assert not (tmp_path / 'evidence').exists()


@pytest.mark.parametrize('state', ['FAILED', 'TIMEOUT', 'COMPLETED', 'CANCELLED'])
def test_terminal_archive_retains_originals_and_missing_failure_evidence(tmp_path, state):
    (tmp_path / 'evidence').mkdir();run = tmp_path / 'runs' / collector.RUN_NAME;run.mkdir(parents=True)
    (run / 'numeric-fixture.log').write_bytes(b'raw diagnostic log\n')
    (run / 'resource-watch.jsonl').write_bytes(b'{"phase":"fault"}\n')
    (run / 'retrieval-golden.json').write_text('{"assets":{"index":{"sha256":"already-computed"}}}')
    (tmp_path / 'scheduler-head').mkdir()
    receipt = dict(job=collector.JOB, token=collector.TOKEN, source=collector.SOURCE,root=str(tmp_path),
                   run=str(run),comment=collector.COMMENT)
    (tmp_path / 'scheduler-head/hardware-submit-receipt.json').write_text(json.dumps(receipt))
    output = tmp_path / 'evidence/new'
    manifest = collector.archive_terminal(tmp_path, output, collector.observe(query(state)), query(state))
    assert manifest['acceptance'] is False and manifest['independent_review_receipt'] is False
    assert manifest['status'] == 'TERMINAL_RAW_ARCHIVED_NOT_ACCEPTED'
    assert (output / 'run/numeric-fixture.log').read_bytes() == b'raw diagnostic log\n'
    assert 'numeric-fixture/rank-7.json' in manifest['missing_run_evidence']
    inventory = {r['relative_path']:r for r in manifest['archive_inventory']}
    assert 'slurm-before.json' in inventory and 'slurm-after.json' in inventory
    assert inventory['run/numeric-fixture.log']['validated_original_copy']
    assert 'already-computed' in (output / 'run/retrieval-golden.json').read_text()
    with pytest.raises(FileExistsError):
        collector.archive_terminal(tmp_path, output, collector.observe(query(state)), query(state))


@pytest.mark.parametrize('comment,end', [('foreign', '2026-10-09T04:00:00'), (None, 'Unknown')])
def test_wrong_original_job_or_incomplete_terminal_is_unverified(comment, end):
    value = collector.observe(query(comment=comment, end=end))
    assert not value['terminal_verified'] and not value['acceptance']
    assert value['issues']


def test_purged_scontrol_can_use_bound_actual_primary_sacct():
    value = collector.observe(query(control_exit=1))
    assert value['terminal_verified'] and not value['acceptance']
    assert value['records']['scontrol']['exit_code'] == 1


def test_symlink_escape_is_retained_as_problem_without_reading_foreign_data(tmp_path):
    (tmp_path / 'evidence').mkdir();run=tmp_path / 'runs' / collector.RUN_NAME;run.mkdir(parents=True)
    foreign=tmp_path / 'foreign-index';foreign.write_bytes(b'NO READ')
    (run / 'retrieval-golden.json').symlink_to(foreign)
    value=collector.archive_terminal(tmp_path,tmp_path / 'evidence/archive',collector.observe(query()),query())
    assert any('noncanonical' in p for p in value['archival_issues'])
    assert 'retrieval-golden.json' in value['missing_run_evidence']
    assert not (tmp_path / 'evidence/archive/run/retrieval-golden.json').exists()


def test_slurm_transition_during_archive_is_not_accepted(tmp_path):
    (tmp_path / 'evidence').mkdir()
    value=collector.archive_terminal(tmp_path,tmp_path / 'evidence/archive',collector.observe(query()),query('RUNNING'))
    assert any('changed during archive' in p for p in value['archival_issues'])
    assert not value['acceptance']


def disabled_accounting(state='COMPLETED'):
    valid=query(state)
    def run(argv):
        if argv[0]=='sacct':return dict(command=argv,exit_code=1,stdout='',stderr='Slurm accounting storage is disabled')
        return valid(argv)
    return run


@pytest.mark.parametrize('state',['COMPLETED','FAILED','TIMEOUT'])
def test_actual_disabled_accounting_uses_strong_scontrol_terminal(state):
    value=collector.observe(disabled_accounting(state))
    assert value['terminal_verified'] and value['primary']['authority']=='scontrol'
    assert not value['acceptance'] and not value['issues']
    assert 'disabled' in value['records']['sacct']['stderr']


def test_disabled_accounting_pending_and_purged_record_are_not_terminal():
    assert not collector.observe(disabled_accounting('PENDING'))['terminal_verified']
    def missing(argv):return dict(command=argv,exit_code=1,stdout='',stderr='disabled' if argv[0]=='sacct'else'Invalid job id specified')
    value=collector.observe(missing)
    assert not value['terminal_verified'] and value['primary']is None and value['issues']


@pytest.mark.parametrize('old,new',[('WorkDir=','OtherDir='),('ReqNodeList=h100-5','ReqNodeList=foreign'),(collector.TOKEN,'foreign-token')])
def test_disabled_accounting_rejects_wrong_control_original_bindings(old,new):
    valid=disabled_accounting()
    def bad(argv):
        value=valid(argv)
        if argv[0]=='scontrol':value['stdout']=value['stdout'].replace(old,new)
        return value
    assert not collector.observe(bad)['terminal_verified']


@pytest.mark.parametrize('end',['garbage','(null)','2026-13-09T04:00:00','2026-10-08T02:00:00'])
def test_invalid_terminal_timestamp_never_verified(end):
    assert not collector.observe(query(end=end))['terminal_verified']


def test_sacct_fallback_refuses_foreign_node():
    valid=query(control_exit=1)
    def wrong(argv):
        v=valid(argv)
        if argv[0]=='sacct':v['stdout']=v['stdout'].replace('|h100-5|','|foreign-node|')
        return v
    assert not collector.observe(wrong)['terminal_verified']


def test_command_timeout_preserves_actual_partial_raw_output(monkeypatch):
    import subprocess
    def timeout(*args,**kwargs):raise subprocess.TimeoutExpired(['scontrol'],45,output=b'partial stdout',stderr=b'partial stderr')
    monkeypatch.setattr(collector.subprocess,'run',timeout)
    result=collector.execute(['scontrol','show','job','-o','3345'])
    assert result['exit_code']is None and result['stdout']=='partial stdout'and result['stderr']=='partial stderr'


def test_unexpected_regular_corpus_is_excluded_without_read_or_hash(tmp_path,monkeypatch):
    (tmp_path / 'evidence').mkdir();run=tmp_path / 'runs' / collector.RUN_NAME;run.mkdir(parents=True)
    corpus=run/'wiki-18.jsonl';corpus.write_bytes(b'not a permitted run artifact')
    original=collector.copy_regular
    def copy(source,destination):
        assert source!=corpus,'corpus must never be read or hashed'
        return original(source,destination)
    monkeypatch.setattr(collector,'copy_regular',copy)
    value=collector.archive_terminal(tmp_path,tmp_path / 'evidence/archive',collector.observe(query()),query())
    assert any('excluded unexpected artifact' in p for p in value['archival_issues'])
    assert not(tmp_path / 'evidence/archive/run/wiki-18.jsonl').exists()
