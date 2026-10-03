"""CPU actual function paths with stub process/lease/GPU roles; NOT HF acceptance."""
import hashlib,io,json,os,sys,types
from pathlib import Path
import pytest
HERE=Path(__file__).parent;sys.path.insert(0,str(HERE/'strict-tools'))
import hf_activation_capture as capture
import hf_launcher as launcher
from training_evidence import COMMIT

def test_actual_capture_source_digest_takes_path_not_str(tmp_path,monkeypatch):
 own={'uid':os.getuid(),'uids':[os.getuid()]*4,'cgroup':'/slurm/job_101/step_0'}
 monkeypatch.setenv('SLURM_JOB_ID','101');monkeypatch.setattr(capture,'process_identity',lambda *args:own);monkeypatch.setattr(capture,'checker_identity',lambda *args:{'launcher':own,'checker':dict(own,pid=2000,process_start_ticks='123'),'checker_sha256':'a'*64});monkeypatch.setattr(capture,'capture_lease',lambda *args:{'scope':'synthetic CPU lease only'})
 audit=capture.capture_hf_activation(1000,tmp_path/'checker','101','GPU-CPU-SYNTHETIC',tmp_path/'lock')
 assert audit['capture_source_sha256']==hashlib.sha256(Path(capture.__file__).read_bytes()).hexdigest()


def test_actual_hf_launcher_update_self_and_preflight_source_sha_cpu_child(tmp_path,monkeypatch):
 # GPU and lease proof interfaces are explicitly stubbed; only child/log/source execution is real.
 source=tmp_path/'repo/search_r1/baselines/preflight.py';source.parent.mkdir(parents=True);source.write_text('# CPU native preflight source fixture')
 model=tmp_path/'checker.py';model.write_text("import time;print('CPU CHECKER ONLY',flush=True);time.sleep(.1)\n");out=tmp_path/'evidence';out.mkdir()
 gpu={'uuid':'GPU-CPU-SYNTHETIC','memory_used':0.,'memory_total':1.,'utilization':0.}
 preflight=types.SimpleNamespace(__file__=str(source),validate_inherited_gpu_leases=lambda **kw:None,gpu_snapshot=lambda:({'0':gpu},set()),check_selected_gpus=lambda *args:[gpu])
 package=types.ModuleType('search_r1');baseline=types.ModuleType('search_r1.baselines');baseline.preflight=preflight;package.baselines=baseline
 monkeypatch.setitem(sys.modules,'search_r1',package);monkeypatch.setitem(sys.modules,'search_r1.baselines',baseline)
 monkeypatch.setenv('SLURM_JOB_ID','101');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0');monkeypatch.setenv('SEARCH_R1_GPU_LEASES',json.dumps([{'uuid':gpu['uuid'],'fd':20}]))
 monkeypatch.setattr(launcher.subprocess,'check_output',lambda args,**kw:COMMIT+'\n' if 'rev-parse' in args else source.read_bytes())
 monkeypatch.setattr(launcher.subprocess,'run',lambda *args,**kw:None)
 original_popen=launcher.subprocess.Popen
 def popen(command,**kw):
  assert set(kw.pop('pass_fds'))=={9,20}
  return original_popen(command,**kw)
 monkeypatch.setattr(launcher.subprocess,'Popen',popen)
 monkeypatch.setattr(launcher,'capture_lease',lambda *args:{'scope':'synthetic CPU lease only'});monkeypatch.setattr(launcher,'capture_hf_activation',lambda *args:{'scope':'synthetic CPU checker process only'})
 written=[];monkeypatch.setattr(launcher,'emit_activation',lambda audit,path:written.append(audit));monkeypatch.setattr(launcher.sys,'stdout',types.SimpleNamespace(buffer=io.BytesIO()))
 record=launcher.run(model,out,tmp_path,hashlib.sha256(model.read_bytes()).hexdigest())
 assert record['launcher_source_sha256']==hashlib.sha256(Path(launcher.__file__).read_bytes()).hexdigest()
 assert record['preflight_source']['sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
 assert written==[record] and (out/'reload.log').read_bytes()==b'CPU CHECKER ONLY\n'
