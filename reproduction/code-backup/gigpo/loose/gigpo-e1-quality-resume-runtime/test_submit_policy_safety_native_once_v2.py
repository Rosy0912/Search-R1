import importlib.util,json,hashlib,subprocess
from pathlib import Path
import pytest

def fixture(tmp_path,monkeypatch,queue='',answer='3300\n',returncode=0):
 p=Path(__file__).with_name('submit_policy_safety_native_once_v2.py')
 spec=importlib.util.spec_from_file_location('once',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 for key in ('CANDIDATE','REVIEW','PHYSICAL','BINDING','INTENT','RECEIPT'):
  monkeypatch.setattr(m,key,tmp_path/(key+'.json'))
 m.CANDIDATE.write_text(json.dumps(dict(status='CANDIDATE_NOT_READY',code_commit='52befe968c856162d1bc79cd0171bb38309f4f9e',allowed_safety_modes=['native_fixture'],native_test_threshold_sets=[[100.,1.]],restore_step=25,expected_optimizer_steps_per_outer=10,restore_checkpoint='healthy25',evidence=[],helpers=str(m.ROOT/'tools/policy-safety-native-52befe9-v2'),resource_guard=dict(path=str(m.ROOT/'tools/policy-safety-native-52befe9-v2/resource_guard_v2.py'),sha256='9644657c579eae63e5a70d0396deb7dfd93ddfd3f38a1604133982b6d195a444'),resource_ownership=dict(path=str(m.ROOT/'tools/policy-safety-native-52befe9-v2/resource_ownership_v2.py'),sha256='7726925b1b02151a681cc1778b71b2c8984dfd87af01512179901d7faf9d8fed'),repo='/new-source')))
 m.PHYSICAL.write_text('{}')
 candidate='3276b298ee324e0d87719d0cf835af2ae2a93b22e4818dc65229decb3053c0d5';physical='c65db5f2df3be25b62d17945efe2a6091410c3bbcb935e036e86bbe64fd88099'
 monkeypatch.setattr(m,'EXPECTED_CANDIDATE_SHA256',candidate)
 monkeypatch.setattr(m,'EXPECTED_PHYSICAL_SHA256',physical)
 old=m.sha
 monkeypatch.setattr(m,'sha',lambda p:candidate if p==m.CANDIDATE else physical if p==m.PHYSICAL else old(p))
 m.REVIEW.write_text(json.dumps(dict(status='PASS',scope='GIGPO_POLICY_SAFETY_NATIVE_ONLY',candidate_sha256=candidate,physical_sha256=physical,submit_helper_sha256=old(p))))
 calls=[]
 monkeypatch.setattr(m.subprocess,'check_output',lambda args,**kw:queue if args[0]=='squeue' else '')
 def submit(args,**kw):
  calls.append(args);return subprocess.CompletedProcess(args,returncode,answer,'')
 monkeypatch.setattr(m.subprocess,'run',submit)
 return m,calls

def test_success_claim_blocks_second_submission(tmp_path,monkeypatch):
 m,calls=fixture(tmp_path,monkeypatch);m.main()
 assert m.INTENT.exists() and m.BINDING.exists() and json.loads(m.RECEIPT.read_text())['job']=='3300'
 with pytest.raises(AssertionError):m.main()
 assert len(calls)==1

def test_uncertain_response_preserves_claim_and_no_retry(tmp_path,monkeypatch):
 m,calls=fixture(tmp_path,monkeypatch,answer='timeout',returncode=1)
 with pytest.raises(AssertionError):m.main()
 assert m.INTENT.exists() and m.RECEIPT.exists() and 'job' not in json.loads(m.RECEIPT.read_text())
 with pytest.raises(AssertionError):m.main()
 assert len(calls)==1

def test_occupied_node_creates_no_submission_claim(tmp_path,monkeypatch):
 m,calls=fixture(tmp_path,monkeypatch,queue='3257 RUNNING')
 with pytest.raises(AssertionError):m.main()
 assert not m.INTENT.exists() and not m.BINDING.exists() and calls==[]
