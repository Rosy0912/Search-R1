from pathlib import Path
import json,hashlib,importlib.util,sys,time,os
S=Path('/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010'); R=S/'group-G_POS/runs'; E=R/'evaluations/outer-15'; D=E/'execution'
def ref(p):
 b=p.read_bytes(); return dict(path=str(p.resolve()),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def j(p,pin=None):
 b=p.read_bytes()
 if pin: assert hashlib.sha256(b).hexdigest()==pin
 return json.loads(b)
def bound(r):return j(Path(r['path']),r['sha256'])
c=j(S/'group-G_POS/receipts/config.json','e7c0659bc6544d3440b7f67b1ba2f976c56c79d8371e23b4dbe614cbb894f661'); gp=Path(c['paths']['helper_root'])/'gates.py'
assert ref(gp)['sha256']=='a8b7659f83380c0808ab589baa386d0f8eadb298f951f525776aed0ece65bef5'
sp=importlib.util.spec_from_file_location('original_gates_cpu_read',gp); g=importlib.util.module_from_spec(sp); sp.loader.exec_module(g)
def live(expected):
 try:
  a=g.process(expected['pid']); assert g.same_process(a,expected) and a['run_id']==expected['run_id'] and a['uid']==1001
  return dict(state='LIVE',pid=a['pid'],uid=a['uid'],starttick=a['starttick'],ppid=a['ppid'],run_id=a['run_id'],wait_returncode='UNKNOWN')
 except (FileNotFoundError,ProcessLookupError):return dict(state='ABSENT',pid=expected['pid'],uid=expected['uid'],starttick=expected['starttick'],run_id=expected['run_id'],wait_returncode='UNKNOWN')
 except Exception as e:return dict(state='IDENTITY_UNKNOWN',pid=expected['pid'],error_type=type(e).__name__,error=str(e),wait_returncode='UNKNOWN')
status=j(R/'status.json'); req=j(D/'request.json','36ff94733134bdba2f0d0fb3750badc9e52b2eea08e0c462bbc6204c1b8375cc')
assert req['completed_outer']==15 and req['run_id']==c['run_id'] and req['variant']=='G_POS'
assert req['input_identity']['raw_checkpoint']['manifest']['sha256']=='fda38c2a8a60fc13512c8a3b29ae2a2c5b4ba02aa732104a02257cc478ba4050'
workers=[]; indices=[]; counts={'hits':0,'final_invalid':0,'valid_gold_mismatch':0}
for rank in range(8):
 lp=D/('worker-%d-launch.json'%rank); wp=D/('worker-%d'%rank)
 row={'rank':rank,'launch_exists':lp.exists(),'complete_exists':(wp/'complete.json').exists(),'primary_rows':0}
 if lp.exists():
  launch=j(lp);row['process']=live(launch['identity']);row['launch']=ref(lp)
 if (wp/'started.json').exists():
  st=j(wp/'started.json');assert st['rank']==rank and st['request']['sha256']==ref(D/'request.json')['sha256'] and st['evaluation_id']==req['evaluation_id']
  assert st['actual_loaded_adapter']['tensor_sha256']==req['input_identity']['expected_adapter_tensor_sha256'];row['loaded_adapter_digest_matches_raw150']=True;row['started']=ref(wp/'started.json')
 own=[];trailing=0
 if (wp/'results.jsonl').exists():
  with (wp/'results.jsonl').open('rb') as f:
   for line in f:
    if not line.endswith(b'\n'):trailing+=1;continue
    rr=json.loads(line);assert rr['rank']==rank and rr['primary'] is True and rr['completed_outer']==15 and rr['evaluation_id']==req['evaluation_id']
    own.append(rr['validation_index']);counts['hits']+=rr['protocol_em'];counts['final_invalid']+=rr['failure_class']=='final_invalid';counts['valid_gold_mismatch']+=rr['failure_class']=='valid_gold_mismatch'
  assert own==list(range(rank,512,8))[:len(own)]
 if (wp/'complete.json').exists():
  cc=j(wp/'complete.json');assert cc['accepted'] and cc['primary_rows']==64 and len(own)==64 and cc['actual_loaded_adapter']['tensor_sha256']==req['input_identity']['expected_adapter_tensor_sha256'];row['complete']=ref(wp/'complete.json')
 row['primary_rows']=len(own);row['trailing_partial_lines']=trailing;indices.extend(own);workers.append(row)
assert len(indices)==len(set(indices))
waves=[]
for n in range(2):
 tp=D/('wave-%d-actual-exit.json'%n);rp=D/('wave-%d-released.json'%n)
 row={'wave':n,'actual_terminal_exists':tp.exists(),'actual_release_exists':rp.exists()}
 if tp.exists():
  tv=j(tp);row['actual_wait_returncodes']=[v['actual_wait_returncode'] for v in tv['child_exit_evidence']];row['terminal']=ref(tp)
 if rp.exists():
  rv=j(rp);assert rv['actual_exit']['sha256']==ref(tp)['sha256'];row['release_accepted']=rv['release']['accepted'] and rv['release']['released'];row['release']=ref(rp)
 waves.append(row)
parent=E/'parent-complete.json';child=E/'child-complete.json';summary=D/'summary.json';finished=False;endpoint={}
if parent.exists() and child.exists() and summary.exists():
 p=j(parent);ch=j(child);su=j(summary);ep=p['exit_proof'];actual=bound(ep['actual_terminal']);assert actual['child_exit_evidence']==ep['child_exit_evidence']
 parentwait=[x['actual_wait_returncode'] for x in ep['child_exit_evidence']];childwait=[x['actual_wait_returncode'] for x in ch['child_exit_evidence']]
 assert p['accepted'] and p['completed_outer']==ch['completed_outer']==su['completed_outer']==15 and ch['accepted'] and su['accepted'] and ch['primary_slots']==su['primary_slots']==len(indices)==512
 assert parentwait==[0] and childwait==[0]*8 and ep['gpu_release']['all_owned_release']['accepted'] and ch['gpu_release']['accepted'] and ch['gpu_release']['released']
 assert ref(child)['sha256']==p['child']['sha256'] and ref(summary)['sha256']==ch['summary']['sha256'] and ch['request']['sha256']==ref(D/'request.json')['sha256']
 assert ch['raw_checkpoint']['manifest']['sha256']==req['input_identity']['raw_checkpoint']['manifest']['sha256'] and p['raw_checkpoint']['sha256']==req['input_identity']['raw_checkpoint']['manifest']['sha256']
 assert sorted(indices)==list(range(512)) and all(x['primary_rows']==64 and x['complete_exists'] for x in workers)
 assert su['overall']['hits']==counts['hits'] and su['overall']['final_invalid']==counts['final_invalid'] and su['overall']['valid_gold_mismatch']==counts['valid_gold_mismatch']
 finished=True;endpoint={'parent':ref(parent),'child':ref(child),'summary':ref(summary),'evaluation_request':ref(D/'request.json'),'raw_checkpoint_native_rich':p['raw_checkpoint'],'actual_parent_wait_returncodes':parentwait,'actual_eight_worker_wait_returncodes':childwait,'actual_parent_full_release':True,'actual_child_full_release':True,'overall':su['overall'],'datasets':su['datasets'],'score_scope':'Stored strict metadata and actual wait/release, independent paired reader not executed'}
binding=j(Path(c['gates']['memory_binding_path']));owner=binding['observer_identity'];report=j(Path(binding['live_report_path']));assert report['observer_identity']==owner and report['run_id']==c['run_id']
observer=dict(process=live(owner),sample_age_seconds=time.time()-report['current']['sample_unix'],samples=report.get('samples'),samples_below_limit=report.get('samples_below_limit'),current_actual_free_mib={x['index']:x['actual_free_mib'] for x in report['current']['devices']},report_path=binding['live_report_path'])
print(json.dumps(dict(schema='opsd-signal-first15-verified-current-v1',checked_unix=time.time(),uid=os.getuid(),phase=status['phase'],controller=live({'pid':1554467,'uid':1001,'starttick':687295549,'run_id':c['run_id']}),evaluator=live(req['controller']),workers=workers,waves=waves,observer=observer,request=ref(D/'request.json'),raw_checkpoint=req['input_identity']['raw_checkpoint']['manifest'],saved_unique_primary_rows=len(indices),formal_denominator_when_complete=512,parent_complete_exists=parent.exists(),child_complete_exists=child.exists(),summary_exists=summary.exists(),complete_endpoint_metadata=finished,endpoint=endpoint,partial_not_a_final_metric=not finished,paired_request_prepared=False,torch_imported='torch' in sys.modules,no_new_gpu_http_signals_or_source_changes=True,no_raw_content_in_stdout=True),sort_keys=True))
