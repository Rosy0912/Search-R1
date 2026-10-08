import ast,hashlib,json,os,sys,threading,time
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parent))
import terminal as t


def setup(tmp_path,monkeypatch):
 for name in ('qa.jsonl','final.safetensors'):
  (tmp_path/name).write_text(('{}\n'*64) if name=='qa.jsonl' else 'fixture raw bound file')
 result=dict(status='COMPLETE64_UNREVIEWED_NO_TRAINING',producer_epoch=time.time(),quality=dict(denominator=64),metrics={'val/qa_count/all':64},qa_sha256=t.sha(tmp_path/'qa.jsonl'),final_sha256=t.sha(tmp_path/'final.safetensors'))
 (tmp_path/'result.json').write_text(json.dumps(result))
 monkeypatch.setattr(t,'process_identity',lambda pid:dict(pid=pid,starttime_ticks=pid,uid=os.getuid(),cgroup=['1:freezer:/slurm_h100-11/uid_0/job_123/step_batch']))
 return result


def held(tmp_path,monkeypatch):
 setup(tmp_path,monkeypatch);receipt=t.facts(tmp_path,30,20,'123');t.durable(tmp_path/'terminal-held.json',dict(receipt,held_epoch=time.time()));return receipt


def test_actual_drain_then_error_inspection_refuses_authorization(tmp_path,monkeypatch):
 held(tmp_path,monkeypatch);errors=[]
 def join():errors.append('actual in-flight failure');return dict(stopped=True)
 with pytest.raises(RuntimeError,match='not drained or failed'):t.authorize(tmp_path,30,20,'123','n',join,errors)
 assert not (tmp_path/'terminal-authorized.json').exists()


def test_authorization_full_result_and_owner_binding(tmp_path,monkeypatch):
 receipt=held(tmp_path,monkeypatch);order=[]
 def join():order.append('joined');return dict(stopped=True)
 out=t.authorize(tmp_path,30,20,'123','n',join,[])
 assert order==['joined'] and out['receipt']==receipt
 assert json.loads((tmp_path/'terminal-authorized.json').read_text())['sampler_errors']==[]
 with pytest.raises(FileExistsError):t.authorize(tmp_path,30,20,'123','n',join,[])


@pytest.mark.parametrize('fault',['qa_drift','result_pid','cgroup','producer_future','partial64','stopped_false'])
def test_actual_bad_terminal_refused(tmp_path,monkeypatch,fault):
 held(tmp_path,monkeypatch)
 if fault=='qa_drift':(tmp_path/'qa.jsonl').write_text('{}\n')
 elif fault=='result_pid':
  path=tmp_path/'terminal-held.json';v=json.loads(path.read_text());v['child_pid']=99;path.write_text(json.dumps(v))
 elif fault=='cgroup':monkeypatch.setattr(t,'process_identity',lambda pid:dict(pid=pid,starttime_ticks=pid,uid=os.getuid(),cgroup=['foreign']))
 elif fault in ('producer_future','partial64'):
  path=tmp_path/'result.json';v=json.loads(path.read_text());v['producer_epoch']=time.time()+100 if fault=='producer_future' else v['producer_epoch'];v['quality']['denominator']=63 if fault=='partial64' else 64;path.write_text(json.dumps(v))
 with pytest.raises(RuntimeError):t.authorize(tmp_path,30,20,'123','n',lambda:dict(stopped=fault!='stopped_false'),[])
 assert not (tmp_path/'terminal-authorized.json').exists()


def test_real_held_engine_reference_and_disk_handshake(tmp_path,monkeypatch):
 setup(tmp_path,monkeypatch);monkeypatch.setenv('SEARCH_R1_EVAL64_SUPERVISOR_PID',str(os.getppid()));monkeypatch.setenv('SLURM_JOB_ID','123');monkeypatch.setenv('SEARCH_R1_EVAL64_TERMINAL_NONCE','nonce')
 destroyed=[]
 class Engine:
  def __del__(self):destroyed.append(True)
 errors=[]
 def child():
  try:t.hold_owner(tmp_path,Engine(),timeout_seconds=2)
  except Exception as e:errors.append(e)
 worker=threading.Thread(target=child);worker.start()
 deadline=time.monotonic()+2
 while not (tmp_path/'terminal-held.json').exists():
  assert time.monotonic()<deadline;time.sleep(.01)
 assert not destroyed and not (tmp_path/'terminal-child-ack.json').exists()
 t.authorize(tmp_path,os.getpid(),os.getppid(),'123','nonce',lambda:dict(stopped=True),[])
 worker.join(2);assert not worker.is_alive() and not errors and destroyed
 assert (tmp_path/'terminal-child-ack.json').is_file()


def test_exact_greedy_generation_and_write_before_engine_barrier():
 root=Path(__file__).parent;old=root.parent/'qwen25-32b-terminal-fix-refreeze'
 a=ast.parse((root/'eval64.py').read_text());b=ast.parse((old/'eval64.py').read_text())
 for name in ['RLHFDataset','GenerationConfig','LLMGenerationManager']:assert name in (root/'eval64.py').read_text()
 for name in ['quality_gate','make_dataset']:
  aa=[n for n in a.body if isinstance(n,ast.FunctionDef) and n.name==name];bb=[n for n in b.body if isinstance(n,ast.FunctionDef) and n.name==name]
  assert ast.dump(aa[0])==ast.dump(bb[0]) if aa else aa==bb
 run=next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name=='run')
 assert ast.unparse(run.body[-1])=='hold_owner(output, engine)'
 assert 'producer_epoch=time.time()' in ast.unparse(run.body[-3])


def test_ack_cannot_forge_other_permit_or_owner(tmp_path,monkeypatch):
 receipt=held(tmp_path,monkeypatch);authorization=t.authorize(tmp_path,30,20,'123','n',lambda:dict(stopped=True),[])
 t.durable(tmp_path/'terminal-child-ack.json',dict(receipt,ack_epoch=time.time(),authorized_sha256=authorization['authorization_sha256']))
 assert t.verify_ack(tmp_path,authorization)['sha256']==t.sha(tmp_path/'terminal-child-ack.json')
 ack=json.loads((tmp_path/'terminal-child-ack.json').read_text());ack['child_pid']=31;(tmp_path/'terminal-child-ack.json').write_text(json.dumps(ack))
 with pytest.raises(RuntimeError):t.verify_ack(tmp_path,authorization)


def test_paused_actual_write_never_publishes_incomplete_final(tmp_path,monkeypatch):
 started=threading.Event();release=threading.Event();errors=[];write=t.os.write
 def paused(fd,data):
  started.set();assert release.wait(2);return write(fd,data)
 monkeypatch.setattr(t.os,'write',paused)
 final=tmp_path/'terminal-held.json'
 def publish():
  try:t.durable(final,dict(actual='complete',rows=64))
  except BaseException as e:errors.append(e)
 worker=threading.Thread(target=publish);worker.start()
 assert started.wait(2)
 try:
  assert not final.exists()
  private=list(tmp_path.glob('.terminal-held.json.pending-*'))
  assert len(private)==1 and private[0].stat().st_size==0
 finally:release.set();worker.join(2)
 assert not worker.is_alive() and not errors
 assert json.loads(final.read_text())==dict(actual='complete',rows=64)
 assert not list(tmp_path.glob('.*.pending-*'))


def test_atomic_conflict_keeps_actual_original_authorization_bytes(tmp_path):
 final=tmp_path/'terminal-authorized.json';original=b'{"old":"authorization"}\n';final.write_bytes(original)
 with pytest.raises(FileExistsError):t.durable(final,dict(new='must not replace'))
 assert final.read_bytes()==original
 assert list(tmp_path.iterdir())==[final]
