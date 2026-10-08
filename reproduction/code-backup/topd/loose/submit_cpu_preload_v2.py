import subprocess,pathlib,json
remote=r'''python3 - <<'INNER'
import hashlib,json,pathlib,subprocess,tarfile,time
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');archive=r/'tools/topd-32b-cpu-preload-v2.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='ffc36a9e7eaddee6ae84669f81cc8918ddde735e5e1be501dd34d1020dcc2ea9'
target=r/'tools/topd-32b-cpu-preload-v2';target.mkdir(exist_ok=True)
with tarfile.open(archive) as t:
 for m in t.getmembers():
  assert m.isfile() and '/' not in m.name and m.name not in ('.','..')
  b=t.extractfile(m).read();p=target/m.name
  if p.exists():assert p.read_bytes()==b
  else:p.write_bytes(b)
for name,digest in json.loads((target/'files.json').read_text())['files'].items():assert hashlib.sha256((target/name).read_bytes()).hexdigest()==digest
candidate=r/'tools/topd-32b-pipeline-candidate';candidate.mkdir(exist_ok=True)
pins={'pipeline.py':'8d7a847daa6ffc84d62c24c0cdb6250c285c8433883fc37fdfe72b6f0254edde','capacity_reference.py':'2d5da3fdd94b637c1625677b302df4df21395ef138f9de2ce484035eab7298ea','native_gate.py':'c945b54294a4e7d222aa6a8c964f2ac8aa5349c0a3d3199532fe0dbba18e95b0'}
for name,digest in pins.items():
 b=(r/'tools/topd-32b-pipeline-native-cpu-v1'/name).read_bytes();assert hashlib.sha256(b).hexdigest()==digest
 p=candidate/name
 if p.exists():assert p.read_bytes()==b
 else:p.write_bytes(b)
assert (r/'assets-root/assets/qwen2.5-3b/tokenizer.json').is_file()
entry=target/'cpu-preload.sbatch';assert hashlib.sha256(entry.read_bytes()).hexdigest()=='d33ddc0b05e5da46b9cc15f7dca5fd08ddc0df7346b194b68261434aa81d4338'
record=r/'logs/topd32b-cpu-preload-v2-submission.json'
if record.exists():
 s=json.loads(record.read_text());assert s['status']=='SUBMITTED',s
else:
 queue=subprocess.check_output(['squeue','-h','-o','%i|%j'],text=True);assert not any(x.split('|',1)[-1]=='topd32b-official-cpu-fp32-identity' for x in queue.splitlines())
 s=dict(status='UNKNOWN_BEFORE_SUBMISSION',entry=str(entry),entry_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),scope='CPU_FP32_OFFICIAL_BASE_IDENTITY_ONLY_NO_QUALITY_NO_GPU_NO_SCORING',epoch=time.time())
 record.write_text(json.dumps(s,indent=2)+'\n')
 out=subprocess.check_output(['sbatch','--parsable',str(entry),str(r/'logs/qwen25-32b-terminal-fixed413-binding.json')],text=True).strip();jid=out.split(';')[0];assert jid.isdigit(),out
 s.update(status='SUBMITTED',job_id=int(jid));record.write_text(json.dumps(s,indent=2)+'\n')
print(json.dumps(s))
INNER'''
p=subprocess.run(['python3','work/remote_topd.py'],input=remote,text=True,capture_output=True)
print(p.stdout,end='');print(p.stderr,end='')
if p.returncode==0:pathlib.Path('outputs/topd-32b-cpu-preload-v2-submission.json').write_text(p.stdout)
raise SystemExit(p.returncode)
