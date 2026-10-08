import importlib.util,pathlib,json,hashlib,base64,gzip,subprocess,pytest
p=pathlib.Path(__file__).with_name('fetch_standard_terminal_evidence_v2.py');s=importlib.util.spec_from_file_location('fetch_terminal',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def row(b,offset=0):
 z=gzip.compress(b,mtime=0);return dict(offset=offset,size=len(b),sha256=m.digest(b),compressed_sha256=m.digest(z),data=base64.b64encode(z).decode())
@pytest.mark.parametrize('change',['offset','rawsha','compressedsha','oversize'])
def test_bad_transport_never_accepted(change):
 r=row(b'a'*10)
 if change=='offset':r['offset']=8
 if change=='rawsha':r['sha256']='0'*64
 if change=='compressedsha':r['compressed_sha256']='0'*64
 if change=='oversize':r=row(b'x'*(m.CHUNK+1))
 with pytest.raises(ValueError):m.decode_chunk(r,0)
def test_running_or_aged_no_terminal_not_complete():
 for raw in ('JobId=42 JobState=RUNNING ExitCode=0:0 NodeList=h100-11','Invalid job id specified'):
  with pytest.raises(ValueError):m.terminal(dict(job_id=42,returncode=0,stdout=raw),42)
def test_real_chunked_full_file_and_remote_stat_drift(tmp_path,monkeypatch):
 root=tmp_path/'own';logs=root/'logs';evidence=logs/'evidence';evidence.mkdir(parents=True)
 term=logs/'terminal.json';result=evidence/'train-result.json';cleanup=evidence/'owned-cleanup.json'
 term.write_text(json.dumps(dict(job_id=42,returncode=0,stdout='JobId=42 JobState=FAILED ExitCode=1:0 NodeList=h100-11')))
 result.write_text(json.dumps(dict(job='42',status='FAILED')))
 cleanup.write_text(json.dumps(dict(cleanup={'actual_cgroup_empty':True},own_processes_remaining=[],own_gpu_records_remaining=[],parent_gpu_records=[],parent_cuda_initialized=False,baseline_identity_missing=[],original_gpu_baseline_missing=[])))
 big=evidence/'own-resource-samples.jsonl';data=b'{"actual":"resource"}\n'*600000;big.write_bytes(data)
 request=dict(job=42,root=str(evidence),terminal=dict(path=str(term),sha256=m.digest(term.read_bytes())),result=dict(path=str(result),sha256=m.digest(result.read_bytes())),cleanup=dict(path=str(cleanup),sha256=m.digest(cleanup.read_bytes())))
 calls=[]
 def send(source,**kw):
  calls.append(source);source=source.replace("ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')","ROOT=pathlib.Path("+repr(str(root))+")")
  return subprocess.run(['python3','-B','-'],input=source.encode(),capture_output=True)
 got=m.fetch(request,tmp_path/'fetched',send)
 assert (tmp_path/'fetched/own-resource-samples.jsonl').read_bytes()==data
 f=next(f for f in got['files'] if f['name']==big.name);assert len(f['chunks'])==2 and f['sha256']==m.digest(data)
 assert all(len(c.encode())<4*1024**2 for c in calls)
 assert got['verified']['status'].startswith('FULL_ORIGINAL')
 old=big.stat();big.write_bytes(data+b'changed')
 import ast
 chunk=next(c for c in calls if "'op': 'chunk'" in c and "own-resource-samples.jsonl'" in c)
 assert send(chunk).returncode!=0
