"""Upload only frozen code/configs; H100 assets are never relayed by this client."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
from datetime import datetime,timezone
import ssh_retained18 as ssh

HERE=Path(__file__).resolve().parent
ROOT='/dataset/shaoshitong/search-r1-rl/tree-grpo'

def main():
    spec=json.loads((HERE/'deployment.json').read_text())
    package=Path(spec['local_package'])
    archive=HERE/'retained18-code-only.tar.gz'
    with tarfile.open(archive,'w:gz') as stream:
        def code_only(item):
            return None if '__pycache__' in Path(item.name).parts or item.name.endswith('.pyc') else item
        stream.add(package,arcname='instruct300/candidates/'+Path(spec['package']).name,filter=code_only)
        for file,target in [('human-authority-v10.json',spec['authority_path']),
                ('direct-release-v10.json',spec['release_path']),
                ('formal-retained18.sbatch',spec['script_path']),
                ('output-storage-verified.json',ROOT+'/.node-migration/output-storage-verified.json')]:
            stream.add(HERE/file,arcname=target[len(ROOT)+1:])
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    code='''import sys,io,hashlib,json,tarfile,subprocess
from pathlib import Path
root=Path(ROOT);destination=Path('/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008')
copy=json.loads((destination/'.node-migration/result.json').read_text())
assert copy['status']=='LOCAL_ASSETS_AND_PARENT_FILES_COPIED_VERIFIED'
assert root.resolve()==destination and not Path(PACKAGE).exists()
raw=sys.stdin.buffer.read(ARCHIVE_BYTES);assert len(raw)==ARCHIVE_BYTES and hashlib.sha256(raw).hexdigest()==ARCHIVE_SHA
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
 for item in archive.getmembers():
  p=Path(item.name)
  assert not p.is_absolute() and '..' not in p.parts and not item.issym() and not item.islnk()
 archive.extractall(root)
package=Path(PACKAGE);m=json.loads((package/'manifest.json').read_text())
assert hashlib.sha256((package/'manifest.json').read_bytes()).hexdigest()==MANIFEST_SHA
for name,expected in m['files'].items():assert hashlib.sha256((package/name).read_bytes()).hexdigest()==expected,name
for name,expected in m['external_files'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
headers=Path('/mnt/shared/workspace/h100-11_from_h100-9/shaoshitong/search-r1-rl/tree-grpo/python-dev')
assert headers.is_dir()
subprocess.run(['rsync','-a',str(headers)+'/',str(root/'python-dev')+'/'],check=True)
assert (root/'python-dev/usr/include/python3.10/Python.h').is_file()
(root/'instruct300/node-move-logs').mkdir(exist_ok=True)
print(json.dumps({'status':'FROZEN_CODE_ONLY_PACKET_DEPLOYED','manifest_sha256':MANIFEST_SHA,'files_verified':len(m['files']),
 'external_files_verified':len(m['external_files']),'archive_sha256':ARCHIVE_SHA,'native_parent_loaded':False,
 'gpu_work_executed':False,'training_submitted':False,'assets_transferred_via_client':False,'python_headers_copied_within_H100':True}))
'''
    for name,value in [('ROOT',ROOT),('PACKAGE',spec['package']),('ARCHIVE_SHA',digest),('ARCHIVE_BYTES',archive.stat().st_size),('MANIFEST_SHA',spec['manifest_sha'])]:
        code=code.replace(name,repr(value))
    try:
        with archive.open('rb') as stream:
            result=subprocess.run(ssh.route('python3 -c '+shlex.quote(code)),stdin=stream,capture_output=True,text=True,timeout=180)
    except subprocess.TimeoutExpired:
        record=dict(observed_at=datetime.now(timezone.utc).isoformat(),status='CODE_UPLOAD_RESPONSE_UNKNOWN_TIMEOUT',
                    timeout_seconds=180,archive_sha256=digest,archive_bytes=archive.stat().st_size,
                    training_submitted=False,follow_up='Read actual own remote package/process state before any retry')
        (HERE.parents[1]/'outputs/tree-fresh200/RETAINED18_PACKET_DEPLOYMENT_20261008.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record));raise SystemExit(1)
    record=dict(observed_at=datetime.now(timezone.utc).isoformat(),returncode=result.returncode,
                stdout=result.stdout,stderr=result.stderr,archive_sha256=digest,archive_bytes=archive.stat().st_size)
    (HERE.parents[1]/'outputs/tree-fresh200/RETAINED18_PACKET_DEPLOYMENT_20261008.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))
    if result.returncode:raise SystemExit(result.returncode)

if __name__=='__main__':main()
