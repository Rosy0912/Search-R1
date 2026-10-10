from pathlib import Path
import os,json,hashlib,time,sys,subprocess,importlib.util
S=Path('/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010');R=S/'group-G_POS/runs'
def ref(p):
 b=p.read_bytes();st=p.stat();return dict(path=str(p.resolve()),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),device=st.st_dev,inode=st.st_ino,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns)
def j(p,pin=None):
 b=p.read_bytes()
 if pin:assert hashlib.sha256(b).hexdigest()==pin
 return json.loads(b)
assert os.getuid()==1001
bp=S/'receipts/terminal-live-binding-515491d.raw.json';b=j(bp,'61ab162ed87242e41e32cb538a2709b5e8234df3ff99af26454ba5a71ecdba28')
gp=S/'source/training-00fec71/opsd_signal_weight_v1/gates.py';sp=importlib.util.spec_from_file_location('original_gates_terminal_recheck',gp);g=importlib.util.module_from_spec(sp);sp.loader.exec_module(g)
current_server=g.process(1100668);assert g.same_process(current_server,b['server']) and current_server['argv']==b['server']['argv'] and current_server['uid']==1001
st=Path(b['socket']['path']).stat();actual_socket=dict(path=str(Path(b['socket']['path']).resolve()),device=st.st_dev,inode=st.st_ino,ctime_ns=st.st_ctime_ns,mode=st.st_mode,uid=st.st_uid);assert actual_socket==b['socket']
fmt='#{session_id}|#{session_name}|#{pane_id}|#{pane_pid}|#{pane_dead}|#{pane_dead_status}|#{pane_start_command}|#{socket_path}|#{pid}'
t=subprocess.run(['tmux','display-message','-p','-t','%45',fmt],capture_output=True,text=True);assert t.returncode==0
vals=t.stdout.rstrip('\n').split('|');keys=['session_id','session_name','pane_id','pane_pid','pane_dead','pane_dead_status','pane_start_command','socket_path','pid'];tm=dict(zip(keys,vals));assert len(vals)==9
assert all(tm[k]==b['tmux'][k] for k in ('session_id','session_name','pane_id','pane_pid','pane_start_command','socket_path','pid'))
assert tm['pane_dead']=='1' and tm['pane_dead_status']=='1'
ro=subprocess.run(['tmux','show-options','-w','-t','%45','remain-on-exit'],capture_output=True,text=True);assert ro.returncode==0 and ro.stdout.strip()=='remain-on-exit on'
seq=[]
for p in sorted((S/'sequence-attempts').glob('*/actual-controller-exit.json')):
 v=j(p)
 if v.get('controller',{}).get('pid')==1554467:
  started=j(p.parent/'started.json');assert started['actual_controller']==v['controller'] and started['supervisor']['pid']==1554337 and started['supervisor']['starttick']==687293806 and started['actual_controller']['starttick']==687295549
  seq.append(dict(reference=ref(p),actual_wait_returncode=v['actual_wait_returncode'],controller=v['controller'],log=v['log'],started=ref(p.parent/'started.json'),original_supervisor=started['supervisor']))
assert len(seq)==1 and seq[0]['actual_wait_returncode']==1
fpath=R/'failure-actual-exits.json';failure=j(fpath);children=failure['children']
summary=dict(total_actual_children=len(children),returncode_counts={str(code):sum(x['actual_wait_returncode']==code for x in children) for code in set(x['actual_wait_returncode'] for x in children)},nonzero_children=[dict(pid=x['identity']['pid'],uid=x['identity']['uid'],starttick=x['identity']['starttick'],run_id=x['identity']['run_id'],actual_wait_returncode=x['actual_wait_returncode']) for x in children if x['actual_wait_returncode']!=0])
driverlog=Path(b['foreground_log']['path']) if isinstance(b['foreground_log'],dict) else Path(b['foreground_log'])
lines=driverlog.read_text(errors='replace').splitlines();trace=[];active=False
for line in lines:
 if line.startswith('Traceback (most recent call last):'):trace=[];active=True
 if active and (line.startswith(('Traceback ','  File ')) or line.startswith('RuntimeError:') or line.startswith('gates.Hold:') or line.startswith('ValueError:')):trace.append(line[:2000])
status=j(R/'status.json');statussmall={k:status.get(k) for k in ('phase','error_type','error','signals_sent','updated_at_unix','variant','run_id','config_sha256')}
locks=[line for line in Path('/proc/locks').read_text().splitlines() if len(line.split())>4 and line.split()[4] in ('1554467','1554337')]
print(json.dumps(dict(schema='opsd-signal-first15-strict-original-driver-terminal-readonly-v1',checked_unix=time.time(),terminal_binding=ref(bp),actual_original_server_identity=current_server,actual_same_socket=actual_socket,actual_same_pane_terminal=tm,original_driver_actual_same_pane_exit_returncode=1,original_controller_actual_supervisor_wait_returncode=1,remain_on_exit=ro.stdout.strip(),actual_sequence_controller_terminal=seq,actual_failure_children_reference=ref(fpath),actual_failure_children_summary=summary,actual_failed_controller_status=dict(reference=ref(R/'status.json'),value=statussmall),original_driver_log_reference=ref(driverlog),driver_filtered_traceback=trace,old_owner_proc_locks=locks,no_new_lock_signal_gpu_task_node_write=True,no_original_worker_wait_inferred=True,torch_imported='torch' in sys.modules),sort_keys=True))
