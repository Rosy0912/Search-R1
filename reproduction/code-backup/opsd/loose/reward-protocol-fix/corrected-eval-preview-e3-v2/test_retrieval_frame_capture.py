"""CPU toy executed-frame capture; no encoder, FAISS, model or HTTP service."""
import json,sys,types
from pathlib import Path
import pytest
HERE=Path(__file__).parent;sys.path.insert(0,str(HERE))
import retrieval_audit_entry as audit
import runtime_binding

def test_real_python_call_frames_observed_without_changing_uvicorn_kwargs(tmp_path,monkeypatch):
 repo=tmp_path/'repo';local=repo/'reproduction/h20/retriever_local.py';server=repo/'search_r1/search/retrieval_server.py';local.parent.mkdir(parents=True);server.parent.mkdir(parents=True)
 local.write_text("import runpy,uvicorn,os\nfrom pathlib import Path\noriginal=uvicorn.run\ndef local_run(app,**kwargs):\n kwargs['host']='127.0.0.1'\n return original(app,**kwargs)\nuvicorn.run=local_run\nrunpy.run_path(str(Path(os.environ['SEARCH_R1_REPO'])/'search_r1/search/retrieval_server.py'),run_name='__main__')\n")
 server.write_text("import uvicorn\nuvicorn.run('CPU-APP',host='not-local-before-native-wrapper',port=18084)\n")
 monkeypatch.setattr(runtime_binding,'load_runtime',lambda *args:({}, {'scope':'synthetic CPU frame test only'}))
 calls=[];uvicorn=types.SimpleNamespace(run=lambda app,**kw:calls.append((app,kw)))
 monkeypatch.setitem(sys.modules,'uvicorn',uvicorn);monkeypatch.setenv('SEARCH_R1_REPO',str(repo));monkeypatch.setenv('SLURM_JOB_ID','101')
 monkeypatch.setattr(audit,'process_identity',lambda pid:{'pid':pid,'scope':'synthetic CPU process fixture only'})
 out=tmp_path/'capture.json';audit.run(local,out,[]);record=json.loads(out.read_text())
 assert calls==[('CPU-APP',{'host':'127.0.0.1','port':18084})]
 assert record['source_frames']=={'search_r1.search.retrieval_server':{'path':str(server),'sha256':audit.digest(server)},'reproduction.h20.retriever_local':{'path':str(local),'sha256':audit.digest(local)}}
 assert record['process']['scope']=='synthetic CPU process fixture only'

def test_wrong_native_target_rejected(tmp_path,monkeypatch):
 monkeypatch.setattr(runtime_binding,'load_runtime',lambda *args:({}, {'scope':'synthetic CPU frame test only'}))
 monkeypatch.setitem(sys.modules,'uvicorn',types.SimpleNamespace(run=lambda **kw:None));monkeypatch.setenv('SEARCH_R1_REPO',str(tmp_path))
 with pytest.raises(ValueError,match='native local'):audit.run(tmp_path/'foreign.py',tmp_path/'out',[])
