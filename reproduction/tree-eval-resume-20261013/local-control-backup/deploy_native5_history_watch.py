"""Deploy one small owned metadata watcher; no training control or tensor reads."""
import argparse,hashlib,json,shlex,subprocess,tarfile,tempfile
from pathlib import Path
import ssh_retained18 as ssh
p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();local=a.packet.resolve();r=json.loads((local/'packet-receipt.json').read_text());release=json.loads((local/'release.json').read_text());remote=r['remote_code_directory'];stage=remote+'.stage'
def call(code):
 x=ssh.run('env CUDA_VISIBLE_DEVICES= python3 -c '+shlex.quote(code),timeout=60)
 if x.returncode:raise RuntimeError(x.stderr[-3000:]+x.stdout[-1000:])
 return json.loads(x.stdout)
prepare=r'''import json,subprocess
from pathlib import Path
r=RECEIPT;release=RELEASE;remote=Path(r['remote_code_directory']);stage=Path(STAGE)
fields=dict(line.split('=',1) for line in subprocess.check_output(['systemctl','show',r['training_unit'],'--property=MainPID,ActiveState,InvocationID'],text=True).splitlines())
assert fields['MainPID']!='0' and fields['ActiveState']=='active' and fields['InvocationID']==r['training_invocation_id']
assert not remote.exists() and not stage.exists() and not (Path('/etc/systemd/system')/r['watcher_unit']).exists()
stage.mkdir();run=Path(release['run_directory']);assert run.is_dir();(run/'output/native-manifest-history').mkdir(exist_ok=True)
print(json.dumps(dict(training=fields)))
'''
for key,value in [('RECEIPT',r),('RELEASE',release),('STAGE',stage)]:prepare=prepare.replace(key,repr(value))
initial=call(prepare)
with tempfile.TemporaryFile() as stream:
 with tarfile.open(fileobj=stream,mode='w:gz') as tar:
  for name,sha in r['files'].items():
   path=local/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==sha;tar.add(path,arcname=name,recursive=False)
 stream.seek(0,2);size=stream.tell();stream.seek(0);subprocess.run(ssh.route('tar -xzf - -C '+shlex.quote(stage)),stdin=stream,check=True)
finish=r'''import json,subprocess,hashlib,time
from pathlib import Path
r=RECEIPT;release=RELEASE;remote=Path(r['remote_code_directory']);stage=Path(STAGE)
for name,sha in r['files'].items():assert hashlib.sha256((stage/name).read_bytes()).hexdigest()==sha,name
assert not remote.exists();stage.rename(remote);history=Path(release['run_directory'])/'output/native-manifest-history'
unitfile=Path('/etc/systemd/system')/r['watcher_unit'];assert not unitfile.exists()
with unitfile.open('xb') as f:f.write((remote/r['watcher_unit']).read_bytes())
fields=dict(line.split('=',1) for line in subprocess.check_output(['systemctl','show',r['training_unit'],'--property=MainPID,ActiveState,InvocationID'],text=True).splitlines());assert fields['ActiveState']=='active' and fields['MainPID']!='0' and fields['InvocationID']==r['training_invocation_id']
with (history/'WATCHER_START_ONCE.json').open('x') as f:json.dump(dict(epoch=time.time(),release_sha256=r['release_sha256'],watcher_unit=r['watcher_unit'],training_observation=fields),f,indent=2)
subprocess.run(['systemctl','daemon-reload'],check=True);started=subprocess.run(['systemctl','start',r['watcher_unit']],capture_output=True,text=True)
result=dict(status='OWN_CODE_WATCHER_START_DISPATCHED',epoch=time.time(),receipt=r,start_returncode=started.returncode,start_stderr=started.stderr,training_signals_sent=False,source_modified=False,watcher_systemd=subprocess.check_output(['systemctl','show',r['watcher_unit'],'--property=MainPID,ActiveState,SubState,Result,InvocationID,CPUUsageNSec,MemoryCurrent'],text=True),training_after=subprocess.check_output(['systemctl','show',r['training_unit'],'--property=MainPID,ActiveState,InvocationID'],text=True))
with (history/'WATCHER_START_RESULT.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result));assert started.returncode==0
'''
for key,value in [('RECEIPT',r),('RELEASE',release),('STAGE',stage)]:finish=finish.replace(key,repr(value))
o=call(finish);o['compressed_code_bytes']=size;o['training_before']=initial['training'];a.output.write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(dict(status=o['status'],epoch=o['epoch'],watcher_systemd=o['watcher_systemd'],training_after=o['training_after'],compressed_code_bytes=size)))
