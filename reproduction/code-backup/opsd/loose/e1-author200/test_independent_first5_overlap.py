"""Independent local CPU counterexamples; no live Slurm, GPU, signature, or transport."""
import importlib.util,json,os,subprocess,sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('independent_overlap',HERE/'first5-overlap-1cfb93790632/first5_overlap.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fake_process(pid,job=None,required=True):
 return {'pid':pid,'uid':0,'start_ticks':'123','present':True,'live':True}

def test_controller_waits_real_child_when_evidence_write_raises(tmp_path,monkeypatch):
 monkeypatch.setattr(m,'pause_context',lambda _: {})
 monkeypatch.setattr(m,'process',fake_process)
 real=m.write_original
 def fail(path,obj):
  if path.name=='srun-start-original.json': raise OSError('Independent evidence write failure')
  return real(path,obj)
 monkeypatch.setattr(m,'write_original',fail)
 marker=tmp_path/'child-finished'
 argv=[sys.executable,'-c','import pathlib,time;time.sleep(.08);pathlib.Path('+repr(str(marker))+').write_text("done")']
 with pytest.raises(OSError,match='evidence write'):m.controller({'command':argv},tmp_path)
 assert marker.read_text()=='done'
 assert not (tmp_path/'srun-exit-original.json').exists()

def test_exclusive_original_marker_has_one_winner_under_parallel_processes(tmp_path):
 target=tmp_path/'exclusive.json'
 script='import importlib.util,pathlib,sys; p=pathlib.Path(sys.argv[1]); s=importlib.util.spec_from_file_location("x",sys.argv[2]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.write_original(p,{"writer":sys.argv[3]})'
 children=[subprocess.Popen([sys.executable,'-c',script,str(target),str(HERE/'first5-overlap-1cfb93790632/first5_overlap.py'),str(i)],stdout=subprocess.PIPE,stderr=subprocess.PIPE) for i in range(2)]
 result=[child.communicate()+(child.returncode,) for child in children]
 assert sorted(x[2] for x in result)==[0,1]
 assert json.loads(target.read_bytes())['writer'] in ('0','1')
 assert any(b'FileExistsError' in x[1] for x in result)

def test_observe_rejects_reused_pid_start_ticks(tmp_path,monkeypatch):
 launch=tmp_path/'launch.json';launch.write_text('{}')
 (tmp_path/'durable-launch-original.json').write_text(json.dumps({'process':{'pid':55,'present':True,'start_ticks':'old'}}))
 monkeypatch.setattr(m,'physical',lambda path,directory=False:Path(path))
 monkeypatch.setattr(m,'process',lambda *a,**kw:{'pid':55,'present':True,'start_ticks':'new'})
 with pytest.raises(ValueError,match='PID reused'):m.observe({'job_id':'3261','output':str(tmp_path),'checkpoint_manifest_sha256':'a'*64},tmp_path,launch,'b'*64)
