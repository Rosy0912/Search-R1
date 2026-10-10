"""Compact read-only actual R progress; never fetch row pairs or model features."""
import datetime,hashlib,json,pathlib,shlex,subprocess,sys
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control')
import control
c=json.loads((HERE/'run-config.json').read_bytes());rid=c['diagnostic_run_id'][:12]
remote=r'''
from pathlib import Path
import datetime,hashlib,json,os,socket,subprocess
out=Path(P_OUT);parent=out.parent
assert socket.gethostname()==P_NODE
rows=[]
for p in sorted(out.glob('TRAIN-*-summary.json')):
    raw=p.read_bytes();d=json.loads(raw)
    assert d['status']=='ACCEPTED_RECONSTRUCTED_R_STEP' and d['trajectory_count']==2560 and d['native_scalar_comparison']['passed']
    rows.append(dict(step=d['step'],N=d['N'],sum_C=d['sum_C'],sum_R=d['sum_R'],sum_R2=d['sum_R2'],sum_CR=d['sum_CR'],individual_pair_Pearson=d['individual_pair_Pearson'],elapsed_seconds=d['elapsed_seconds'],native_scalar_comparison=d['native_scalar_comparison'],summary_sha256=hashlib.sha256(raw).hexdigest(),output_sha256=d['output_sha256']))
rows.sort(key=lambda r:r['step'])
assert [r['step'] for r in rows]==list(range(1,len(rows)+1))
pane=subprocess.run(['tmux','list-panes','-t',P_SESSION,'-F','#{pane_id} #{pane_pid} #{pane_dead} #{pane_dead_status}'],capture_output=True,text=True)
result=dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),node=socket.gethostname(),complete_R_iterations=len(rows),complete_paired_rows=2560*len(rows),iterations=rows,pane=pane.stdout.strip(),pane_query_rc=pane.returncode,raw_pairs_not_exported=True)
for name in ['status.json','inputs.json']:
    p=out/name
    if p.is_file():result[name]=json.loads(p.read_bytes())
for p in sorted(out.glob('probe-loaded-*.json')):
    result.setdefault('loaded_versions',[]).append(json.loads(p.read_bytes()))
memory=out/'memory.jsonl'
if memory.is_file():
    with memory.open('rb') as f:
        f.seek(max(0,memory.stat().st_size-20000)); tail=f.read().splitlines()
    if tail:
        try:result['latest_memory']=json.loads(tail[-1])
        except ValueError:result['latest_memory']='concurrent_partial'
stdout=parent/('logs/R-'+P_RID+'.stdout.jsonl')
if stdout.is_file() and stdout.stat().st_size:
    with stdout.open('rb') as f:
        f.seek(max(0,stdout.stat().st_size-10000)); lines=f.read().splitlines()
    try:result['latest_stdout_summary']=json.loads(lines[-1])
    except ValueError:result['latest_stdout_summary']='concurrent_partial'
stderr=parent/('logs/R-'+P_RID+'.stderr.log')
result['private_stderr_bytes']=stderr.stat().st_size if stderr.is_file() else 0
print(json.dumps(result))
'''
for tag,val in [('P_OUT',c['output']),('P_NODE',c['expected_node']),('P_SESSION','g1-full-pair-R-'+rid),('P_RID',rid)]:remote=remote.replace(tag,repr(val))
r=subprocess.run(control.route('baai-a800',shlex.join(['python3','-c',remote])),capture_output=True,timeout=45)
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');p=HERE/('progress-'+stamp)
p.with_suffix('.stdout.json').write_bytes(r.stdout);p.with_suffix('.stderr.txt').write_bytes(r.stderr)
p.with_suffix('.execution.json').write_text(json.dumps(dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rc=r.returncode,source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
assert r.returncode==0,'progress snapshot failed; do not infer training failure'
d=json.loads(r.stdout)
print(json.dumps(dict(complete_R_iterations=d['complete_R_iterations'],complete_paired_rows=d['complete_paired_rows'],pane=d['pane'],status=d.get('status.json'),latest_stdout_summary=d.get('latest_stdout_summary'),private_stderr_bytes=d['private_stderr_bytes'],actual_memory=d.get('latest_memory',{}).get('candidate') if isinstance(d.get('latest_memory'),dict) else None)))
