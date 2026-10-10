"""Published remote CPU math reference; exports only small moments/counts."""
import argparse
import ast
import base64
import datetime
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
PREFIX = 'experiments/search_e1/control/g1_reward_correlation_20261010/'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--publication', required=True)
    args = parser.parse_args()
    pub = json.loads(pathlib.Path(args.publication).read_bytes())
    rows = {x['path']: x['sha256'] for x in pub['files']}
    encoded = {}
    for name in ['normalized_advantage_moments.py', 'test_normalized_advantage_moments.py',
                 'read_normalized_advantage_moments.py', 'pairwise_order_concordance.py']:
        raw = (HERE / name).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert rows[PREFIX + name] == digest
        encoded[name] = dict(sha256=digest, data=base64.b64encode(raw).decode())
    core = pathlib.Path('/Users/shaoshitong/Documents/Search-G1/repo/verl/trainer/ppo/core_algos.py')
    tree = ast.parse(core.read_bytes())
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'compute_grpo_outcome_advantage')
    ast_sha = hashlib.sha256(ast.dump(fn, include_attributes=False).encode()).hexdigest()
    assert ast_sha == '3289eeb5b20239288e4f863f6f97b0fec593f1b6c748e8b291d7a0dc8f11f777'
    g = json.loads((HERE / 'selected8-group-distributions-accepted-compact.json').read_bytes())
    cfg = json.loads((HERE / 'statistics-config-prepared-v2.json').read_bytes())
    payload = dict(node=cfg['expected_node'], python=cfg['original_base'] + '/env/bin/python',
        summary=g['full_summary_remote_path'], summary_sha=g['full_summary_sha256'],
        groups_sha=g['raw_group_artifact_sha256'], helper=encoded['normalized_advantage_moments.py'],
        group_helper=encoded['pairwise_order_concordance.py'],
        native_core=cfg['original_base'] + '/repo/verl/trainer/ppo/core_algos.py', native_ast_sha=ast_sha,
        commit=pub['source_commit'])
    remote = r'''
import base64,hashlib,json,sys
j=json.load(sys.stdin)
raw=base64.b64decode(j['helper']['data']);group_raw=base64.b64decode(j['group_helper']['data'])
assert hashlib.sha256(raw).hexdigest()==j['helper']['sha256']
assert hashlib.sha256(group_raw).hexdigest()==j['group_helper']['sha256']
sys.argv=['published-normalized-advantage','--summary',j['summary'],'--summary-sha256',j['summary_sha'],
    '--groups-sha256',j['groups_sha'],'--expected-node',j['node'],'--helper-sha256',j['helper']['sha256'],
    '--group-helper-sha256',j['group_helper']['sha256'],'--native-core-algos',j['native_core'],
    '--native-grpo-ast-sha256',j['native_ast_sha']]
exec(compile(raw,'<published-normalized-advantage>','exec'),dict(__name__='__main__',
    __published_helper_bytes__=raw,__bound_group_helper_bytes__=group_raw))
'''
    sys.path.insert(0, '/Users/shaoshitong/Documents/remote_codex_control')
    import control
    command = ['env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1', payload['python'], '-B', '-c', remote]
    result = subprocess.run(control.route('baai-a800', shlex.join(command)),
        input=json.dumps(payload).encode(), capture_output=True, timeout=100)
    prefix = HERE / ('normalized-advantage-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    prefix.with_suffix('.stdout.json').write_bytes(result.stdout)
    prefix.with_suffix('.stderr.txt').write_bytes(result.stderr)
    prefix.with_suffix('.execution.json').write_text(json.dumps(dict(actual_rc=result.returncode,
        publication=pub['source_commit'], expected_native_grpo_AST_sha256=ast_sha)) + '\n')
    assert result.returncode == 0, 'preserve actual failure; inspect read-only result'
    print(result.stdout.decode())


if __name__ == '__main__':
    main()
