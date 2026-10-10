"""Read actual complete iterations / retained pane rc; no model or remote writes."""
import datetime,hashlib,json,pathlib,shlex,subprocess,sys
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control')
import control
cfg=json.loads((HERE/'raw-C-run-config.json').read_bytes())
session='g1-full-raw-C-'+cfg['diagnostic_run_id'][:12]
remote=r'''
from pathlib import Path
import hashlib,json,os,socket,subprocess,datetime
p=Path(P_BASE); assert socket.gethostname()==P_NODE
rows=[]
for q in sorted((p/'raw-C').glob('TRAIN-*-summary.json')):
    raw=q.read_bytes(); d=json.loads(raw)
    assert d['native_EM_check_pass'] is True and d['trajectory_count']==2560
    rows.append(dict(step=d['step'],C_hits=d['C_hits_exact'],invalid=d['invalid_final_action_count'],file_sha256=hashlib.sha256(raw).hexdigest()))
rows.sort(key=lambda x:x['step'])
assert [r['step'] for r in rows]==list(range(1,len(rows)+1))
pane=subprocess.run(['tmux','list-panes','-t',P_SESSION,'-F','#{pane_id} #{pane_pid} #{pane_dead} #{pane_dead_status}'],capture_output=True,text=True)
st=[]
for q in (p/'logs').glob('*'):
    st.append(dict(name=q.name,bytes=q.stat().st_size))
result=dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),node=socket.gethostname(),uid=os.getuid(),complete_iterations=len(rows),complete_C_pairs=len(rows)*2560,iterations=rows,pane=pane.stdout.strip(),pane_query_rc=pane.returncode,log_sizes=st,shaped_R_replayed=False,Pearson_computed=False)
accepted=p/'raw-C/accepted-summary.json'
stdout=p/'logs/raw-C.stdout.json'
if accepted.is_file():
    raw=accepted.read_bytes(); result['accepted_summary_sha256']=hashlib.sha256(raw).hexdigest();result['accepted_status']=json.loads(raw)['status']
if stdout.is_file() and stdout.stat().st_size:
    result['exit_stdout_summary']=json.loads(stdout.read_bytes())
print(json.dumps(result))
'''
for tag,val in [('P_BASE',cfg['remote_base']),('P_NODE',cfg['expected_node']),('P_SESSION',session)]: remote=remote.replace(tag,repr(val))
r=subprocess.run(control.route('baai-a800',shlex.join(['python3','-c',remote])),capture_output=True,timeout=45)
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
prefix=HERE/('raw-C-progress-'+stamp)
prefix.with_suffix('.stdout.json').write_bytes(r.stdout);prefix.with_suffix('.stderr.txt').write_bytes(r.stderr)
prefix.with_suffix('.execution.json').write_text(json.dumps(dict(rc=r.returncode,observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),helper_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
assert r.returncode==0,'progress read failed; no inference of training failure'
d=json.loads(r.stdout)
print(json.dumps(dict(complete_iterations=d['complete_iterations'],complete_C_pairs=d['complete_C_pairs'],pane=d['pane'],accepted_status=d.get('accepted_status'),exit_status=d.get('exit_stdout_summary',{}).get('status'),reason=d.get('exit_stdout_summary',{}).get('reason'))))
