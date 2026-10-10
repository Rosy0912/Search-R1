"""Root-invoked deployment, preparation and execution of selected8 CPU statistics.

All row-level inputs and outputs remain on the original node. This controller
transfers published code/configuration and compact audit summaries only. Import
and local fixtures do not contact a node. Root must publish a prepared config
before invoking execute. No generation, GPU allocation or process signals.
"""
import argparse
import base64
import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PREFIX = 'experiments/search_e1/control/g1_reward_correlation_20261010/'
SELECTED = [1, 20, 40, 60, 80, 100, 120, 140]
PAYLOAD = ['selected_pair_statistics.py', 'pairwise_pearson.py',
           'test_selected_pair_statistics.py', 'test_pairwise_pearson.py']
C_HELPER_SHA = 'c9385df75c06f3c7a28e41e0ae0b502866fa3418e54d3162ec0a7a2996732c34'
OLD_CONFIG_SHA = '3e8b03f02d95225bf8afdfc6886fd281568942237e2c1e024262af5ae017bc13'
OLD_MANIFEST_SHA = 'de2464fd4d419a5f17e3aec5fe056edaf56803b11c3e4fc23d1201cdee8bd7a5'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def original_context(cohort_path):
    cohort_path = Path(cohort_path)
    co = json.loads(cohort_path.read_bytes())
    assert co['selected_steps'] == SELECTED and co['final_pair_count'] == 20480
    assert re.fullmatch('[0-9a-f]{32}', co['cohort_id'])
    assert co['original_run_id'] == 'db474b6abf654ad9892b02ae67ab6f86'
    D = Path(co['remote_parent']).parent
    assert str(D).startswith('/data/shaoshitong/Search-G1-reward-reconstruction-20261010-')
    tag = co['cohort_id'][:12]
    assert Path(co['remote_source']).parent == D
    assert Path(co['remote_parent']).name == 'parallel-cohort-' + tag
    paths = dict(diagnostic_base=str(D), source=str(D / ('statistics-source-' + tag)),
                 output=str(D / ('statistics-selected-' + tag)),
                 C_directory=str(D / 'raw-C'), C_helper_path=str(D / 'source/reconstruct_train_raw_c_preflight_v2.py'))
    payload_sha = {name: digest((HERE / name).read_bytes()) for name in PAYLOAD}
    common, producers = None, []
    covered = []
    for entry in co['entries']:
        cfg_path = Path(entry['local_config']); raw = cfg_path.read_bytes(); cfg = json.loads(raw)
        assert cfg['gpu_index'] == entry['gpu'] and cfg['selected_steps'] == entry['steps']
        assert cfg['cohort_id'] == co['cohort_id'] and cfg['expected_run_id'] == co['original_run_id']
        assert cfg['base'] == co['base'] and cfg['expected_node'] == co['expected_node']
        assert cfg['C_base'] == paths['C_directory'] and cfg['C_helper_path'] == paths['C_helper_path']
        assert cfg['C_helper_sha256'] == C_HELPER_SHA
        identity = {k: cfg[k] for k in ['run_manifest_sha256', 'resolved_config_sha256', 'reward_constants']}
        assert common is None or common == identity
        common = identity
        covered += entry['steps']
        remote_cfg = str(Path(co['remote_source']) / 'configs' / ('gpu' + str(entry['gpu']) + '.json'))
        producers.append(dict(selected_steps=entry['steps'], config_path=remote_cfg,
            config_sha256=digest(raw), source_directory=co['remote_source'],
            source_manifest_path=str(Path(co['remote_source']) / 'source-manifest.json'),
            source_manifest_sha256=digest((cohort_path.parent / 'source-manifest.json').read_bytes()),
            input_path=str(Path(cfg['output']) / 'inputs.json')))
    assert sorted(covered) == SELECTED[1:] and len(covered) == len(set(covered))
    old_source = D / 'R-source-d0e94bdf64d8'
    producers.insert(0, dict(selected_steps=[1], config_path=str(old_source / 'run-config.json'),
        config_sha256=OLD_CONFIG_SHA, source_directory=str(old_source),
        source_manifest_path=str(old_source / 'source-manifest.json'), source_manifest_sha256=OLD_MANIFEST_SHA,
        input_path=str(D / 'R-d0e94bdf64d8/inputs.json')))
    return dict(cohort=co, paths=paths, payload_sha256=payload_sha, common=common, producers=producers)


def context(cohort_path, recovery_path=None):
    ctx = original_context(cohort_path)
    ctx['blueprint_paths'] = [str(Path(cohort_path).resolve()),
        str((Path(cohort_path).parent / 'source-manifest.json').resolve())]
    original = ctx['cohort']
    entries = original['entries']
    if recovery_path is None:
        ctx['blueprint_paths'] += [x['local_config'] for x in entries]
        return ctx
    recovery_path = Path(recovery_path).resolve()
    rec = json.loads(recovery_path.read_bytes())
    assert rec['selected_steps'] == SELECTED and rec['final_pair_count'] == 20480
    assert rec['recovery_only_steps'] == [60,80,120,140]
    assert rec['retained_original_parallel_steps'] == [20,40,100]
    assert rec['recovery_from_preserved_parallel_cohort'] == original['cohort_id']
    for key in ['base','expected_node','original_run_id']:
        assert rec[key] == original[key]
    D = Path(ctx['paths']['diagnostic_base'])
    assert Path(rec['remote_parent']).parent == D and Path(rec['remote_source']).parent == D
    assert re.fullmatch('[0-9a-f]{32}', rec['cohort_id'])
    assert Path(rec['remote_parent']).name == 'parallel-recovery-' + rec['cohort_id'][:12]
    # Preserve already accepted original producers; never point at failed runs.
    retained = [x for x in ctx['producers'] if x['selected_steps'] in [[1],[20,40],[100]]]
    assert sorted(s for x in retained for s in x['selected_steps']) == [1,20,40,100]
    ctx['blueprint_paths'] += [x['local_config'] for x in entries if x['steps'] in [[20,40],[100]]]
    ctx['blueprint_paths'] += [str(recovery_path), str(recovery_path.parent / 'source-manifest.json')]
    recovered, coverage = [], []
    for entry in rec['entries']:
        cfg_path = Path(entry['local_config']); raw = cfg_path.read_bytes(); cfg = json.loads(raw)
        assert cfg_path.resolve() == cfg_path and cfg_path.is_relative_to(HERE)
        assert cfg['gpu_index'] == entry['gpu'] and cfg['selected_steps'] == entry['steps']
        assert cfg['cohort_id'] == rec['cohort_id'] and cfg['expected_run_id'] == original['original_run_id']
        assert cfg['base'] == original['base'] and cfg['expected_node'] == original['expected_node']
        assert cfg['C_base'] == ctx['paths']['C_directory'] and cfg['C_helper_path'] == ctx['paths']['C_helper_path']
        assert cfg['C_helper_sha256'] == C_HELPER_SHA
        assert {k: cfg[k] for k in ['run_manifest_sha256','resolved_config_sha256','reward_constants']} == ctx['common']
        coverage += entry['steps'];ctx['blueprint_paths'].append(str(cfg_path))
        source = Path(rec['remote_source'])
        recovered.append(dict(selected_steps=entry['steps'],config_path=str(source/'configs'/('gpu'+str(entry['gpu'])+'.json')),
            config_sha256=digest(raw),source_directory=str(source),source_manifest_path=str(source/'source-manifest.json'),
            source_manifest_sha256=digest((recovery_path.parent/'source-manifest.json').read_bytes()),
            input_path=str(Path(cfg['output'])/'inputs.json')))
    assert sorted(coverage) == [60,80,120,140] and len(coverage) == len(set(coverage))
    ctx['producers'] = retained + recovered
    assert sorted(s for x in ctx['producers'] for s in x['selected_steps']) == SELECTED
    ctx['paths']['source'] += '-v2'
    ctx['paths']['output'] += '-v2'
    ctx['recovery_cohort_id'] = rec['cohort_id']
    return ctx


def published(publication_path, names):
    pub = json.loads(Path(publication_path).read_bytes())
    rows = {r['path']: r['sha256'] for r in pub['files']}
    for name in names:
        assert rows[PREFIX + name] == digest((HERE / name).read_bytes()), 'Unpublished or changed local bytes: ' + name
    return pub


def published_cohort(pub, ctx):
    """Pin exact mixed producer blueprints and manifests to publication."""
    rows = {r['path']: r['sha256'] for r in pub['files']}
    for name in ctx['blueprint_paths']:
        p = Path(name)
        assert p.resolve() == p and p.is_relative_to(HERE)
        relative = p.relative_to(HERE).as_posix()
        assert rows[PREFIX+relative] == digest(p.read_bytes()), 'Unpublished producer blueprint: '+relative


REMOTE_COMMON = r'''
import base64,datetime,hashlib,json,os,pathlib,re,socket,subprocess,sys
j=json.load(sys.stdin);ctx=j['context'];co=ctx['cohort'];P=ctx['paths'];D=pathlib.Path(P['diagnostic_base']);S=pathlib.Path(P['source'])
assert sys.platform.startswith('linux') and socket.gethostname()==co['expected_node']
assert os.getuid()==D.stat().st_uid and D.resolve()==D and D.is_dir()
assert S.parent==D and S.resolve()==S
def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def metadata(path,expected=None):
    p=pathlib.Path(path);assert p.resolve()==p and p.is_file() and not p.is_symlink() and p.stat().st_size<=32*1024*1024
    raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();assert expected is None or h==expected
    return json.loads(raw),h
assert sha(P['C_helper_path'])==j['C_helper_sha256']
def cpu_env():
    e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTHONPATH=str(pathlib.Path(P['C_helper_path']).parent),TMPDIR=str(S/'fixture-tmp'))
    return e
'''

REMOTE_DEPLOY = REMOTE_COMMON + r'''
assert not S.exists();S.mkdir(mode=0o700);(S/'fixture-tmp').mkdir()
records=[]
for row in j['files']:
    q=S/row['name'];assert q.parent==S and q.resolve()==q and not q.exists()
    raw=base64.b64decode(row['base64']);assert hashlib.sha256(raw).hexdigest()==row['sha256']
    with q.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    assert sha(q)==row['sha256'];records.append(dict(name=row['name'],bytes=len(raw),sha256=row['sha256']))
r=subprocess.run([co['base']+'/env/bin/python','-B','-m','unittest','discover','-s',str(S),'-p','test_*.py'],env=cpu_env(),capture_output=True,timeout=180)
with (S/'fixture.stdout.txt').open('xb') as f:f.write(r.stdout)
with (S/'fixture.stderr.txt').open('xb') as f:f.write(r.stderr)
text=r.stderr.decode(errors='replace')
accepted=r.returncode==0 and re.search(r'Ran 77 tests\b',text) is not None and '\nOK\n' in text and 'skipped' not in text
print(json.dumps(dict(phase='DEPLOY_CPU_FIXTURES',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=str(S),node=socket.gethostname(),uid=os.getuid(),files=records,CPU_fixture_rc=r.returncode,CPU_fixtures=77,zero_skip=accepted,accepted=accepted,CPU_fixture_summary=text[-1800:],GPU_started=False,original_producers_unmodified=True)))
raise SystemExit(0 if accepted else 2)
'''

REMOTE_PREPARE = REMOTE_COMMON + r'''
assert S.is_dir()
for name,h in ctx['payload_sha256'].items():assert sha(S/name)==h
run,runsha=metadata(pathlib.Path(co['base'])/'logs/run-manifest.json',ctx['common']['run_manifest_sha256'])
assert run['run_id']==co['original_run_id'] and run['execution_host']==socket.gethostname()
ci,csha=metadata(pathlib.Path(P['C_directory'])/'inputs.json')
assert ci['run_id']==co['original_run_id'] and ci['helper_sha256']==j['C_helper_sha256']
producers=[]
for item in ctx['producers']:
    cfg,cfgsha=metadata(item['config_path'],item['config_sha256']);mf,msha=metadata(item['source_manifest_path'],item['source_manifest_sha256'])
    for name,h in mf['source_sha256'].items():assert sha(pathlib.Path(item['source_directory'])/name)==h
    assert cfg['expected_run_id']==co['original_run_id'] and cfg['base']==co['base'] and cfg['C_base']==P['C_directory']
    inp,isha=metadata(item['input_path'])
    assert inp['original_run_id']==co['original_run_id'] and inp['config_sha256']==cfgsha and inp['source_manifest_sha256']==msha and inp['raw_online_R_saved'] is False
    producers.append(dict(selected_steps=item['selected_steps'],config_path=item['config_path'],config_sha256=cfgsha,source_directory=item['source_directory'],source_manifest_path=item['source_manifest_path'],source_manifest_sha256=msha,inputs_sha256=isha))
print(json.dumps(dict(phase='PREPARE_COMPACT_BINDINGS_ONLY',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),C_inputs_sha256=csha,R_sources=producers,producer_rows_not_read=True,raw_records_not_exported=True,GPU_started=False,original_producers_unmodified=True)))
'''

REMOTE_EXECUTE = REMOTE_COMMON + r'''
assert S.is_dir()
for name,h in ctx['payload_sha256'].items():assert sha(S/name)==h
cfg_raw=base64.b64decode(j['config_base64']);assert hashlib.sha256(cfg_raw).hexdigest()==j['config_sha256'];cfg=json.loads(cfg_raw)
assert cfg['selected_steps']==[1,20,40,60,80,100,120,140] and cfg['diagnostic_base']==str(D) and cfg['output_new']==P['output']
assert not pathlib.Path(cfg['output_new']).exists()
# Check every selected receipt before executing the consumer. No waiting or
# inference from producer initialization/status; the consumer revalidates all rows.
selected=[]
for item in cfg['R_sources']:
    rc,rsha=metadata(item['config_path'],item['config_sha256']);inp,isha=metadata(pathlib.Path(rc['output'])/'inputs.json',item['inputs_sha256'])
    assert inp['original_run_id']==cfg['native_run_id'] and inp['config_sha256']==rsha
    for step in item['selected_steps']:
        s,_=metadata(pathlib.Path(rc['output'])/('TRAIN-'+str(step)+'-summary.json'))
        assert s['status']=='ACCEPTED_RECONSTRUCTED_R_STEP' and s['step']==step and s['N']==s['trajectory_count']==2560
        assert s['native_scalar_comparison']['passed'] is True and s['original_online_row_rewards_available'] is False
        selected.append(step)
assert sorted(selected)==cfg['selected_steps'] and len(selected)==len(set(selected))==8
target=S/'config-selected8.json'
if target.exists():assert sha(target)==j['config_sha256']
else:
    with target.open('xb') as f:f.write(cfg_raw);f.flush();os.fsync(f.fileno())
r=subprocess.run([co['base']+'/env/bin/python','-B',str(S/'selected_pair_statistics.py'),'--config',str(target),'--config-sha256',j['config_sha256'],'--helper-sha256',ctx['payload_sha256']['selected_pair_statistics.py']],env=cpu_env(),capture_output=True,timeout=300)
token=j['invocation_id']
for suffix,raw in [('stdout.json',r.stdout),('stderr.txt',r.stderr)]:
    with (S/('execution-'+token+'.'+suffix)).open('xb') as f:f.write(raw)
for name,h in ctx['payload_sha256'].items():assert sha(S/name)==h
assert r.returncode==0,'CPU consumer HOLD; preserve diagnostics and do not overwrite'
result=json.loads(r.stdout);assert result['status']=='ACCEPTED_SELECTED_TRAIN_STEPS_PAIR_CORRELATION' and result['total_individual_pairs']==20480
report,h=metadata(result['report_remote_path'],result['report_sha256'])
assert report['selected_steps']==cfg['selected_steps'] and report['total_pairs']==20480 and report['original_online_R_available'] is False
keys=['step','N','sumC','sumR','sumR2','sumCR','C_mean','R_mean','Pearson']
compact=dict(phase='ACCEPTED_SELECTED8_CPU_STATISTICS',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),CPU_consumer_rc=r.returncode,selected_steps=report['selected_steps'],total_individual_pairs=report['total_pairs'],per_step=[{k:s[k] for k in keys} for s in report['steps']],selected_step_pooled=report['pooled'],report_remote_path=result['report_remote_path'],report_sha256=h,config_sha256=j['config_sha256'],raw_records_not_exported=True,selected_only_not_dense_1_to_150=True,historical_R_reconstructed_not_online_bitwise=True,original_producers_unmodified=True,GPU_started=False)
encoded=json.dumps(compact,allow_nan=False).encode();assert len(encoded)<=30*1024
print(encoded.decode())
'''


def remote_call(code, payload, phase, timeout):
    sys.path.insert(0, '/Users/shaoshitong/Documents/remote_codex_control')
    import control
    invocation = stamp()
    payload['invocation_id'] = invocation
    original_python = payload['context']['cohort']['base'] + '/env/bin/python'
    command = ['env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1', original_python, '-B', '-c', code]
    r = subprocess.run(control.route('baai-a800', shlex.join(command)),
                       input=canonical(payload), capture_output=True, timeout=timeout)
    root = HERE / ('statistics-control-' + phase + '-' + invocation)
    for suffix, raw in [('stdout.json', r.stdout), ('stderr.txt', r.stderr)]:
        with root.with_suffix('.' + suffix).open('xb') as f:
            f.write(raw)
    with root.with_suffix('.execution.json').open('xb') as f:
        f.write(canonical(dict(rc=r.returncode, phase=phase, invocation_id=invocation,
            observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            controller_sha256=digest(Path(__file__).read_bytes()))))
    assert r.returncode == 0, 'Preserve failed/HOLD receipt; inspect actual state before another explicit action'
    value = json.loads(r.stdout)
    print(json.dumps(value, ensure_ascii=False))
    return value


def prepare_config(ctx, observed):
    P, co = ctx['paths'], ctx['cohort']
    assert observed['phase'] == 'PREPARE_COMPACT_BINDINGS_ONLY'
    assert sorted(step for item in observed['R_sources'] for step in item['selected_steps']) == SELECTED
    return dict(expected_node=co['expected_node'], native_run_id=co['original_run_id'],
        original_base=co['base'], diagnostic_base=P['diagnostic_base'], selected_steps=SELECTED,
        run_manifest_sha256=ctx['common']['run_manifest_sha256'], C_directory=P['C_directory'],
        C_inputs_sha256=observed['C_inputs_sha256'], C_helper_path=P['C_helper_path'],
        C_helper_sha256=C_HELPER_SHA, pairwise_source_path=str(Path(P['source']) / 'pairwise_pearson.py'),
        pairwise_source_sha256=ctx['payload_sha256']['pairwise_pearson.py'],
        R_sources=observed['R_sources'], output_new=P['output'],
        semantics='selected8 actual per-trajectory C_raw versus R_reconstructed; no dense coverage or original online R claim',
        per_row_outputs_remote_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['deploy', 'prepare', 'execute'], required=True)
    parser.add_argument('--publication', required=True)
    parser.add_argument('--cohort-config', default=str(HERE / 'historical_replay/parallel_selected/cohort-config.json'))
    parser.add_argument('--recovery-cohort-config', default=None)
    parser.add_argument('--prepared-config', default=str(HERE / 'statistics-config-prepared-v2.json'))
    args = parser.parse_args()
    names = ['statistics_control_v2.py'] + PAYLOAD
    if args.phase == 'execute':
        assert Path(args.prepared_config).resolve() == HERE / 'statistics-config-prepared-v2.json'
        names += ['statistics-config-prepared-v2.json']
    pub = published(args.publication, names)
    ctx = context(args.cohort_config, args.recovery_cohort_config)
    published_cohort(pub, ctx)
    payload = dict(context=ctx, C_helper_sha256=C_HELPER_SHA, source_commit=pub['source_commit'])
    if args.phase == 'deploy':
        payload['files'] = [dict(name=name, sha256=ctx['payload_sha256'][name],
            base64=base64.b64encode((HERE / name).read_bytes()).decode()) for name in PAYLOAD]
        remote_call(REMOTE_DEPLOY, payload, 'deploy', 230)
    elif args.phase == 'prepare':
        output = Path(args.prepared_config)
        assert output.resolve() == HERE / 'statistics-config-prepared-v2.json' and not output.exists()
        observed = remote_call(REMOTE_PREPARE, payload, 'prepare', 90)
        cfg = prepare_config(ctx, observed)
        with output.open('xb') as f:
            f.write(canonical(cfg))
        print(json.dumps(dict(prepared_config=str(output), prepared_config_sha256=digest(output.read_bytes()),
            requires_publication_before_execute=True, statistics_not_executed=True, raw_records_not_exported=True)))
    else:
        cfg_raw = Path(args.prepared_config).read_bytes(); cfg = json.loads(cfg_raw)
        assert cfg['output_new'] == ctx['paths']['output'] and cfg['selected_steps'] == SELECTED
        payload.update(config_base64=base64.b64encode(cfg_raw).decode(), config_sha256=digest(cfg_raw))
        remote_call(REMOTE_EXECUTE, payload, 'execute', 350)


if __name__ == '__main__':
    main()
