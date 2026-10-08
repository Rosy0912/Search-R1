"""Deploy only new files and amend the existing, unstarted student job once."""
import base64
import hashlib
import io
import json
import pathlib
import subprocess
import tarfile

WORK = pathlib.Path(__file__).resolve().parent
files = [WORK / name for name in ['bind_topd_instruct300_v8.py', 'start_topd_teacher85_approved.py']]
files += [p for p in (WORK / 'topd-instruct300-launch-v6').rglob('*')
          if p.is_file() and '__pycache__' not in p.parts]
manifest = {str(p.relative_to(WORK)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
buffer = io.BytesIO()
with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
    for p in files:
        archive.add(p, arcname=str(p.relative_to(WORK)))
body = r'''import pathlib,json,hashlib,base64,tarfile,io,subprocess,os,time
R=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
record=R/'logs/topd3224-teacher85-amendment.json'
if record.exists():
 print(json.dumps({'existing_record':json.loads(record.read_text()),'queue':subprocess.check_output(['scontrol','show','job','3224','-o'],text=True),'no_repeat_mutation':True}));raise SystemExit(0)
query=subprocess.check_output(['scontrol','show','job','3224','-o'],text=True)
assert 'JobId=3224 ' in query and 'JobState=PENDING ' in query and 'StartTime=Unknown ' in query,query
manifest=MANIFEST
for name,expected in manifest.items():
 p=R/'tools'/name
 if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==expected, str(p)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)),mode='r:gz') as archive:
 for member in archive.getmembers():
  assert member.isfile() and not member.name.startswith('/') and '..' not in pathlib.PurePosixPath(member.name).parts
 archive.extractall(R/'tools')
for name,expected in manifest.items():assert hashlib.sha256((R/'tools'/name).read_bytes()).hexdigest()==expected,name
entry=R/'tools/start_topd_after_teacher3218.py';old=entry.read_bytes()
backup=R/'tools/start_topd_after_teacher3218.teacher100-original.py'
if not backup.exists():backup.write_bytes(old)
assert backup.read_bytes()==old
stub="import runpy\nrunpy.run_path('/dataset/shaoshitong/search-r1-rl/topd/tools/start_topd_teacher85_approved.py',run_name='__main__')\n"
rec={'status':'PREPARED_NOT_RELEASED','epoch':time.time(),'job':3224,'teacher_checkpoint':85,'teacher_original_job':3228,'original_dispatcher_sha256':hashlib.sha256(old).hexdigest(),'new_files':manifest,'no_new_submission':True,'original_scontrol':query}
with record.open('x') as f:json.dump(rec,f,indent=2);f.flush();os.fsync(f.fileno())
tmp=entry.with_suffix('.teacher85.tmp');tmp.write_text(stub);os.replace(tmp,entry)
result=subprocess.run(['scontrol','update','JobId=3224','Dependency=0'],capture_output=True,text=True,timeout=20)
rec.update(status='DEPENDENCY_UPDATED' if result.returncode==0 else 'DEPENDENCY_UPDATE_FAILED',returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,updated_dispatcher_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),fresh_scontrol=subprocess.check_output(['scontrol','show','job','3224','-o'],text=True))
tmp=record.with_suffix('.tmp');tmp.write_text(json.dumps(rec,indent=2)+'\n');os.replace(tmp,record);print(json.dumps(rec));raise SystemExit(result.returncode)
'''.replace('MANIFEST', repr(manifest)).replace('PAYLOAD', repr(base64.b64encode(buffer.getvalue()).decode()))
command = "env -u LD_LIBRARY_PATH python3 - <<'DEPLOY'\n" + body + "\nDEPLOY\n"
result = subprocess.run(['python3', str(WORK / 'remote_topd_retained18.py')],
                        input=command, text=True, capture_output=True, timeout=180)
out = WORK.parent / 'outputs/topd3224-teacher85-amendment-result.log'
out.write_text(result.stdout + result.stderr)
print(result.stdout)
print(result.stderr)
raise SystemExit(result.returncode)
