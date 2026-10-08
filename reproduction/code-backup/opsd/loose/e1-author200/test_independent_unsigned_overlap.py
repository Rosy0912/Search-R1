"""Independent exact unsigned wrapper CPU fixtures. No remote or approval work."""
import copy,importlib.util,json,hashlib,types
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('unsigned_independent',HERE/'first5-unsigned-overlap-317dec4d4dd1/unsigned_receipt_overlap.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
s=importlib.util.spec_from_file_location('old_independent',HERE/'first5-overlap-1cfb93790632/first5_overlap.py');b=importlib.util.module_from_spec(s);s.loader.exec_module(b)

def completion_fixture(tmp_path,monkeypatch):
 monkeypatch.setattr(m,'base',lambda:types.SimpleNamespace(unique=b.unique,physical=lambda p,directory=False:Path(p)))
 cfg={'job_id':'3261','pause_nonce':'a'*64,'checkpoint_manifest_sha256':'b'*64}
 payload=dict(status='PASS_INDEPENDENT_OPSD_E1_STEP5_REVIEW',source_commit=m.SOURCE,job_id='3261',pause_nonce=cfg['pause_nonce'],checkpoint_manifest_sha256=cfg['checkpoint_manifest_sha256'])
 file=tmp_path/'unsigned-step5-payload.json';file.write_text(json.dumps(payload))
 records={x+'-exit-original.json':{'returncode':0} for x in ('controller','srun','producer')}
 records['unsigned-preparation-completed-review-required.json']={'scope':m.SCOPE,'source_commit':m.SOURCE,'job_id':'3261','checkpoint_manifest_sha256':cfg['checkpoint_manifest_sha256'],'cuda_is_initialized':False,'payload':{'path':str(file),'sha256':m.hashfile(file)}}
 records['cpu-step-original.json']={'source_commit':m.SOURCE,'job_id':'3261','checkpoint_manifest_sha256':cfg['checkpoint_manifest_sha256'],'cuda_visible_devices':'','cuda_is_initialized':False,'omp_num_threads':'1','process':{'uid':0,'cgroup_original':'1:freezer:/slurm_h100-10/uid_0/job_3261/step_2\n'}}
 for name,v in records.items():(tmp_path/name).write_text(json.dumps(v))
 return cfg,records,file

def test_exact_immutable_tool_and_helper_join():
 p=HERE/'first5-unsigned-overlap-317dec4d4dd1';r=json.loads((p/'manifest.json').read_bytes())
 assert hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest()=='317dec4d4dd1ce2b2a535d37b0cf39543422fbfff93e7d322e7aabf5cdb29319'
 assert hashlib.sha256((p/'unsigned_receipt_overlap.py').read_bytes()).hexdigest()==r['tools']['unsigned_receipt_overlap.py']=='4ca065ea1c8ac49f510c8a8a45225b4cce09075c84aff96d8659382045380d59'
 assert r['overlap_script_sha256']==hashlib.sha256((HERE/'first5-overlap-1cfb93790632/first5_overlap.py').read_bytes()).hexdigest()
 assert r['producer_sha256']==hashlib.sha256((HERE/'first5-12424d01025e/receipt_payload.py').read_bytes()).hexdigest()

def test_unsigned_completion_requires_three_real_zero_exits_and_exact_cpu_identity(tmp_path,monkeypatch):
 cfg,records,payload=completion_fixture(tmp_path,monkeypatch);assert m.completion(cfg,tmp_path)
 for name in ('controller','srun','producer'):
  path=tmp_path/(name+'-exit-original.json');path.write_text(json.dumps({'returncode':False}))
  with pytest.raises(ValueError):m.completion(cfg,tmp_path)
  path.write_text(json.dumps({'returncode':7}));assert m.completion(cfg,tmp_path) is False
  path.write_text(json.dumps({'returncode':0}))
 cpu=tmp_path/'cpu-step-original.json';original=records[cpu.name]
 for key,value in [('cuda_visible_devices','0'),('cuda_is_initialized',True),('omp_num_threads','8'),('job_id','3210')]:
  bad=dict(original,**{key:value});cpu.write_text(json.dumps(bad))
  with pytest.raises(ValueError):m.completion(cfg,tmp_path)
 cpu.write_text(json.dumps(original));assert m.completion(cfg,tmp_path)

def test_postproducer_payload_mutation_cannot_be_promoted_to_ready(tmp_path,monkeypatch):
 cfg,records,payload=completion_fixture(tmp_path,monkeypatch);payload.write_bytes(payload.read_bytes()+b'\n')
 with pytest.raises(ValueError,match='changed'):m.completion(cfg,tmp_path)
