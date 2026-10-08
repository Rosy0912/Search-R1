import base64,hashlib,io,json,pathlib,tarfile,types,pytest
import remote_bootstrap as b
import native_entry as n
import capture_cpu_binding as c

def archive(name,files):
 records={k:{'size':len(v),'sha256':hashlib.sha256(v).hexdigest()} for k,v in files.items()};m=json.dumps({'files':records}).encode();all_files={**files,'files.json':m};buf=io.BytesIO()
 with tarfile.open(fileobj=buf,mode='w:gz') as tar:
  for k,value in all_files.items():
   item=tarfile.TarInfo(name+'/'+k);item.size=len(value);tar.addfile(item,io.BytesIO(value))
 raw=buf.getvalue();return {'name':name,'archive_sha256':hashlib.sha256(raw).hexdigest(),'manifest_sha256':hashlib.sha256(m).hexdigest(),'base64':base64.b64encode(raw).decode()}

def test_actual_archive_installs_exact_and_idempotent(tmp_path,monkeypatch):
 monkeypatch.setattr(b,'ROOT',tmp_path);(tmp_path/'tools').mkdir();p=archive('source-tools',{'entry.py':b'print(1)\n'})
 first=b.install(p);assert b.install(p)==first and (tmp_path/'tools/source-tools/entry.py').read_bytes()==b'print(1)\n'
 assert first['archive_sha256']==p['archive_sha256']

def test_existing_source_mutation_never_overwritten(tmp_path,monkeypatch):
 monkeypatch.setattr(b,'ROOT',tmp_path);(tmp_path/'tools').mkdir();p=archive('source-tools',{'entry.py':b'print(1)\n'});b.install(p)
 target=tmp_path/'tools/source-tools/entry.py';target.write_bytes(b'old mutation')
 with pytest.raises(ValueError,match='never overwrite'):b.install(p)
 assert target.read_bytes()==b'old mutation'

def test_directory_alias_rejected_before_source_write(tmp_path,monkeypatch):
 monkeypatch.setattr(b,'ROOT',tmp_path);(tmp_path/'tools/source-tools').mkdir(parents=True);outside=tmp_path/'outside';outside.mkdir();(tmp_path/'tools/source-tools/alias').symlink_to(outside,target_is_directory=True)
 p=archive('source-tools',{'alias/new.py':b'print(1)\n'})
 with pytest.raises(ValueError):b.install(p)
 assert list(outside.iterdir())==[]

def actual_logs_fixture(tmp_path,monkeypatch):
 monkeypatch.setattr(c,'ROOT',tmp_path);out=tmp_path/'logs/cpu';out.mkdir(parents=True)
 logs={}
 for name,count in [('pipeline',184),('known_tools',14)]:
  p=out/(name+'.log');p.write_text(str(count)+' passed in 2.1s\n');logs[name]={'path':str(p),'sha256':n.sha(p),'passed':count,'returncode':0}
 pending={'directory':str(tmp_path/'tools/g1-production-pipeline-v11'),'manifest_sha256':c.PIPELINE_SHA,'native_expected':184,'native_cpu':{'job_id':9,'passed':184,'log':{'path':logs['pipeline']['path'],'sha256':logs['pipeline']['sha256']},'terminal':None}}
 (out/'pipeline-source-binding.pending.json').write_text(json.dumps(pending));(out/'receipt.json').write_text(json.dumps({'status':'CPU_TESTS_COMPLETE_TERMINAL_PENDING','job_id':9,'logs':logs}));return out

def test_actual_completed_terminal_creates_bound_descriptor(tmp_path,monkeypatch):
 out=actual_logs_fixture(tmp_path,monkeypatch);seen=[]
 def run(argv,**kw):seen.append(argv);return types.SimpleNamespace(returncode=0,stdout='JobId=9 JobState=COMPLETED ExitCode=0:0 NodeList=h100-11',stderr='')
 proof=c.finalize(out,runner=run);assert seen==[['scontrol','show','job','-dd','9']]
 terminal=proof['native_cpu']['terminal'];raw=json.loads(pathlib.Path(terminal['path']).read_text());assert raw['final_state_observed'] and raw['job_id']==9
 assert c.finalize(out,runner=lambda *a,**k:pytest.fail('known receipt must not rerun'))==proof

def test_running_or_wrong_job_never_creates_success_descriptor(tmp_path,monkeypatch):
 out=actual_logs_fixture(tmp_path,monkeypatch)
 with pytest.raises(RuntimeError):c.finalize(out,runner=lambda *a,**kw:types.SimpleNamespace(returncode=0,stdout='JobId=10 JobState=RUNNING ExitCode=0:0 NodeList=h100-11',stderr=''))
 assert not (out/'pipeline-source-binding.json').exists()
 assert json.loads(next(out.glob('scontrol-captured-*.json')).read_text())['final_state_observed'] is False
