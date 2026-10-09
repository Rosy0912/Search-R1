"""Control-only setup of a separate verified build-source HF migration group."""
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import termios

BASE = Path(__file__).resolve().parents[1]
E = BASE / 'outputs/evidence'
spec = importlib.util.spec_from_file_location('ctl', '/Users/shaoshitong/Documents/remote_codex_control/control.py')
ctl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ctl)

def campus(host, command, **kwargs):
    args = ['ssh', '-T', '-oBatchMode=yes', '-oStrictHostKeyChecking=yes', '-oConnectTimeout=8', host, command]
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=12, **kwargs)
        if r.returncode == 0:
            return r
    except subprocess.TimeoutExpired:
        pass
    # Previously authorized per-connection recovery, only after current alias failure.
    args = ['ssh', '-T', '-B', 'en0', '-oProxyCommand=none', '-oProxyJump=none', '-oBatchMode=yes',
            '-oStrictHostKeyChecking=yes', '-oConnectTimeout=8', host, command]
    return subprocess.run(args, capture_output=True, text=True, timeout=45, **kwargs)

def deploy(host, root, files):
    encoded = {name:base64.b64encode(data).decode() for name,data in files.items()}
    code = ('from pathlib import Path;import base64,json;root=Path(' + repr(root) + ');'
            'assert not root.exists(),"Existing group must be inspected";root.mkdir(parents=True);'
            'files=' + repr(encoded) + '\n'
            'for name,data in files.items():(root/name).write_bytes(base64.b64decode(data))\n'
            'print("DEPLOYED")')
    command = 'python3 -c ' + shlex.quote(code)
    r = campus(host, command) if host in ('hkust2','hkust3') else subprocess.run(ctl.route(host,command),capture_output=True,text=True,timeout=45)
    assert r.returncode == 0, host + ': ' + r.stderr[-500:]

def start(host, root, script, token=None):
    code = ('import subprocess,sys,json;from pathlib import Path;root=Path(' + repr(root) + ');'
            'f=(root/' + repr(script+'.log') + ').open("ab");'
            'p=subprocess.Popen([sys.executable,str(root/' + repr(script) + ')],'
            'stdin=subprocess.PIPE,stdout=f,stderr=f,start_new_session=True);'
            + ('p.stdin.write(sys.stdin.readline().encode());' if token else '') +
            'p.stdin.close();print(json.dumps({"pid":p.pid,"script":' + repr(script) + '}))')
    command = 'python3 -c ' + shlex.quote(code)
    r = campus(host, command, input=(token+'\n') if token else '') if host in ('hkust2','hkust3') else subprocess.run(ctl.route(host,command),input=(token+'\n') if token else '',capture_output=True,text=True,timeout=45)
    assert r.returncode == 0, host + ': ' + r.stderr[-500:]
    return json.loads(r.stdout)

def main():
    source = json.loads((E/'native-x-hpc-flash-source-preparation-20261009.json').read_text())['archive-verified.json']
    assert source['status']=='SOURCE_ARCHIVE_AND_FILES_SHA_VERIFIED' and source['files']==6368
    plan = json.dumps({'repo':'Sutoonq/TMP_MODEL','records':[source]}).encode()
    work = BASE/'work'
    read = lambda name:(work/name).read_bytes()
    manifest = (E/'native-x-hpc-flash-build-source-manifest.jsonl').read_bytes()
    assert hashlib.sha256(manifest).hexdigest()==source['manifest_sha256']
    mgmt='/home/lichen/cosearch-flash-build-20261009'
    hk2='/data/shaoshitong/cosearch-flash-build-20261009'
    hk3='/backup/shaoshitong/cosearch-flash-build-20261009'
    hpc='/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/migration-flash-build-20261009'
    deploy('mgmt-host',mgmt,{'plan.json':plan,'cosearch_hf_stream_upload.py':read('cosearch_hf_stream_upload.py')})
    deploy('hkust2',hk2,{'plan.json':plan,'download-network.json':json.dumps({'endpoint':'https://hf-mirror.com','proxy':None}).encode(),
        'cosearch_hf_download_hkust2.py':read('cosearch_hf_download_hkust2.py'),'cosearch_flash_forward_hkust2.py':read('cosearch_flash_forward_hkust2.py')})
    deploy('hkust3',hk3,{'plan.json':plan,'cosearch_flash_forward_hkust3.py':read('cosearch_flash_forward_hkust3.py')})
    deploy('hpc3',hpc,{'manifest.jsonl':manifest,'cosearch_flash_hpc_receiver.py':read('cosearch_flash_hpc_receiver.py')})
    # Terminal echo is disabled before the credential is supplied; no disk copy.
    settings=termios.tcgetattr(sys.stdin.fileno());quiet=settings.copy();quiet[3]&=~termios.ECHO
    termios.tcsetattr(sys.stdin.fileno(),termios.TCSANOW,quiet)
    print('READY_FOR_TASK_CREDENTIAL',flush=True)
    try:token=sys.stdin.readline().strip()
    finally:termios.tcsetattr(sys.stdin.fileno(),termios.TCSANOW,settings)
    assert token
    result={'source':source,'upload':start('mgmt-host',mgmt,'cosearch_hf_stream_upload.py',token),
            'download':start('hkust2',hk2,'cosearch_hf_download_hkust2.py',token),
            'forward2':start('hkust2',hk2,'cosearch_flash_forward_hkust2.py'),
            'forward3':start('hkust3',hk3,'cosearch_flash_forward_hkust3.py'),'mac_data_relay':False}
    token=None
    (E/'native-x-hpc-flash-route-controllers-20261009.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
