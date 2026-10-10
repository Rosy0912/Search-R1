from pathlib import Path
import json,hashlib,os,time,sys,importlib.util,subprocess
S=Path('/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010');R=S/'group-G_POS/runs';E=R/'evaluations/outer-15';D=E/'execution'
def ref(p):
 b=p.read_bytes();st=p.stat();return dict(path=str(p.resolve()),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),device=st.st_dev,inode=st.st_ino,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns)
def j(p,pin=None):
 b=p.read_bytes()
 if pin:assert hashlib.sha256(b).hexdigest()==pin
 return json.loads(b)
def bound(r):return j(Path(r['path']),r['sha256'])
assert os.getuid()==1001
C=S/'group-G_POS/receipts/config.json';c=j(C,'e7c0659bc6544d3440b7f67b1ba2f976c56c79d8371e23b4dbe614cbb894f661')
gr=Path(c['paths']['helper_root']);sp=importlib.util.spec_from_file_location('original_gates_final_readonly',gr/'gates.py');g=importlib.util.module_from_spec(sp);sp.loader.exec_module(g)
def proc(pin):
 try:
  a=g.process(pin['pid']);assert g.same_process(a,pin);return dict(state='LIVE',pid=a['pid'],uid=a['uid'],starttick=a['starttick'],ppid=a['ppid'],run_id=a['run_id'],argv=a['argv'])
 except (FileNotFoundError,ProcessLookupError):return dict(state='ABSENT',expected=pin)
 except Exception as ex:return dict(state='UNKNOWN',pid=pin['pid'],error_type=type(ex).__name__,error=str(ex))
bindingpath=S/'receipts/terminal-live-binding-515491d.raw.json';binding=j(bindingpath,'61ab162ed87242e41e32cb538a2709b5e8234df3ff99af26454ba5a71ecdba28')
# Only small native binding metadata; no payloads or output rows.
binding_top={k:v for k,v in binding.items() if k not in ('groups','group','configs','study','config','scientific_contract','scoped_processes')}
tm=subprocess.run(['tmux','display-message','-p','-t','%45','#{session_id}|#{session_name}|#{pane_id}|#{pane_pid}|#{pane_dead}|#{pane_dead_status}|#{pane_start_command}|#{socket_path}|#{pid}'],capture_output=True,text=True)
opt=subprocess.run(['tmux','show-options','-p','-t','%45','remain-on-exit'],capture_output=True,text=True)
st=j(R/'status.json'); ss=j(S/'study-status.json') if (S/'study-status.json').exists() else None
failedpath=R/'failure-actual-exits.json';failed=j(failedpath) if failedpath.exists() else None
terminal=[]
for p in sorted((R/'phase-exits').glob('*.json')):
 v=j(p)
 if any(x.get('identity',{}).get('pid')==1715934 for x in v.get('child_exit_evidence',[])):terminal.append(dict(reference=ref(p),value=v))
native_req=j(D/'request.json','36ff94733134bdba2f0d0fb3750badc9e52b2eea08e0c462bbc6204c1b8375cc')
proofs=[]
for rank in range(8):
 p=D/('worker-%d'%rank); comp=j(p/'complete.json')
 proofs.append(dict(rank=rank,launch=ref(D/('worker-%d-launch.json'%rank)),started=ref(p/'started.json'),complete=ref(p/'complete.json'),primary_rows=comp['primary_rows'],loaded_adapter_sha256=comp['actual_loaded_adapter']['tensor_sha256'],results=comp.get('results'),trace_refs={k:v for k,v in comp.items() if k in ('generation_trace','http_trace','control','log','exceptions')}))
devices,apps=g.inventory();dev=[d for d in devices if d['index'] in [4,5,6,7]];uuids={d['uuid'] for d in dev};ownapps=[a for a in apps if a['gpu_uuid'] in uuids]
assert [d['uuid'] for d in dev]==c['owner']['gpu_uuids']
oldlocks=[line for line in Path('/proc/locks').read_text().splitlines() if len(line.split())>4 and line.split()[4] in ('1554467','1554337')]
sources={name:ref(gr/name) for name in ('evaluation.py','evaluation_worker.py','gates.py','controller.py','source-manifest.json')}
scriptman=Path(c['helper_source_manifest']['path']);sm=j(scriptman,c['helper_source_manifest']['sha256'])
assert all(sources[k]['sha256']==sm['files'][k]['sha256'] for k in sources if k!='source-manifest.json')
raw=native_req['input_identity']['raw_checkpoint']['manifest'];actual_raw=ref(Path(raw['path']));assert actual_raw['sha256']==raw['sha256']
minimal_status={k:st.get(k) for k in ('schema','run_id','variant','phase','error','error_type','signals_sent','updated_at_unix','config_sha256','outer_index','optimizer_updates')}
sup=proc(dict(pid=1554337,uid=1001,starttick=687293806));ctr=proc(dict(pid=1554467,uid=1001,starttick=687295549))
# study-status is metadata only; report its original actual group waits/driver terminal.
out=dict(schema='opsd-signal-first15-failed-parent-terminal-resource-readonly-v1',checked_unix=time.time(),uid=os.getuid(),supervisor=sup,controller=ctr,tmux_display=dict(actual_returncode=tm.returncode,stdout=tm.stdout.strip(),stderr=tm.stderr.strip()),tmux_remain_on_exit=dict(actual_returncode=opt.returncode,stdout=opt.stdout.strip(),stderr=opt.stderr.strip()),terminal_live_binding=ref(bindingpath),terminal_binding_metadata=binding_top,study_status=dict(reference=ref(S/'study-status.json'),value=ss) if ss else None,failed_controller_status=dict(reference=ref(R/'status.json'),value=minimal_status),failure_actual_exits=dict(reference=ref(failedpath),value=failed) if failed else None,parent_phase_terminal=terminal,original_request=ref(D/'request.json'),raw_checkpoint_actual_manifest=actual_raw,saved_input_identity=native_req['input_identity'],input_weight_payloads_new_full_SHA_read=False,selected_original_sources_fresh=sources,worker_proofs=proofs,devices=dev,compute_apps_on_owned_uuids=ownapps,only_expected_oracle_gpu_process=all(a['pid']==1749337 for a in ownapps),old_controller_or_supervisor_proc_locks=oldlocks,original_parent_log=ref(E/'controller.log'),original_worker_log_refs=[ref(D/('worker-%d.log'%rank)) for rank in range(8)],no_new_gpu_tasks_locks_signals_or_node_writes=True,torch_imported='torch' in sys.modules)
print(json.dumps(out,sort_keys=True))
