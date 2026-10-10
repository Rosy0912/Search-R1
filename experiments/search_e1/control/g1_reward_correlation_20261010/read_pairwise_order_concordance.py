"""Root-invoked published remote CPU read, exporting small aggregates only."""
import argparse, base64, datetime, hashlib, json, pathlib, shlex, subprocess, sys

HERE=pathlib.Path(__file__).resolve().parent
PREFIX='experiments/search_e1/control/g1_reward_correlation_20261010/'

def main():
    p=argparse.ArgumentParser();p.add_argument('--publication',required=True);a=p.parse_args()
    pub=json.loads(pathlib.Path(a.publication).read_bytes());rows={x['path']:x['sha256'] for x in pub['files']}
    names=['pairwise_order_concordance.py','test_pairwise_order_concordance.py','read_pairwise_order_concordance.py']
    helper=None
    for name in names:
        raw=(HERE/name).read_bytes();sha=hashlib.sha256(raw).hexdigest();assert rows[PREFIX+name]==sha
        if name=='pairwise_order_concordance.py':helper=dict(sha256=sha,data=base64.b64encode(raw).decode())
    g=json.loads((HERE/'selected8-group-distributions-accepted-compact.json').read_bytes())
    cfg=json.loads((HERE/'statistics-config-prepared-v2.json').read_bytes())
    remote=r'''
import base64,hashlib,json,sys
j=json.load(sys.stdin);raw=base64.b64decode(j['helper']['data']);assert hashlib.sha256(raw).hexdigest()==j['helper']['sha256']
sys.argv=['published-pairwise-order','--summary',j['summary'],'--summary-sha256',j['summary_sha'],'--groups-sha256',j['groups_sha'],'--expected-node',j['node'],'--helper-sha256',j['helper']['sha256']]
exec(compile(raw,'<published-pairwise-order>','exec'),dict(__name__='__main__',__published_helper_bytes__=raw))
'''
    payload=dict(node=cfg['expected_node'],python=cfg['original_base']+'/env/bin/python',summary=g['full_summary_remote_path'],summary_sha=g['full_summary_sha256'],groups_sha=g['raw_group_artifact_sha256'],helper=helper,commit=pub['source_commit'])
    sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
    command=['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1',payload['python'],'-B','-c',remote]
    r=subprocess.run(control.route('baai-a800',shlex.join(command)),input=json.dumps(payload).encode(),capture_output=True,timeout=100)
    prefix=HERE/('pairwise-order-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    prefix.with_suffix('.stdout.json').write_bytes(r.stdout);prefix.with_suffix('.stderr.txt').write_bytes(r.stderr)
    prefix.with_suffix('.execution.json').write_text(json.dumps(dict(actual_rc=r.returncode,publication=pub['source_commit']))+'\n')
    assert r.returncode==0,'preserve actual failure; inspect the read-only result'
    print(r.stdout.decode())

if __name__=='__main__':main()
