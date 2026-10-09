"""Authorized HKUST3 control bridge; private credential never enters arguments."""
import importlib.util,subprocess,re,shlex,sys,json
from pathlib import Path
spec=importlib.util.spec_from_file_location('setup',Path(__file__).with_name('start_cosearch_flash_hf.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def run(command,payload=''):
 secret=Path('/Users/shaoshitong/Documents/remote_codex_control/memories/hpc3-credentials.private.md').read_text()
 password=re.split(r'密码\s*[:：]?\s*',next(l for l in secret.splitlines() if l.startswith('账号 mou176')),1)[1].strip().strip('`');secret=None
 code="import os,sys,subprocess;env=os.environ.copy();env['SSHPASS']=sys.stdin.readline().rstrip('\\n');data=sys.stdin.read();r=subprocess.run(['sshpass','-e','ssh','-T','-oStrictHostKeyChecking=yes','-oHostKeyAlias=hpc3login.hpc.hkust-gz.edu.cn','-oUserKnownHostsFile=/backup/shaoshitong/topd-hf-migration-20261009/hpc3-known-hosts','-oConnectTimeout=8','mou176@10.120.81.30',"+repr(command)+"],input=data,env=env,capture_output=True,text=True,timeout=60);print(r.stdout,end='');print(r.stderr,file=sys.stderr,end='');sys.exit(r.returncode)"
 return m.campus('hkust3','python3 -c '+shlex.quote(code),input=password+'\n'+payload)
if __name__=='__main__':
 d=json.loads(sys.stdin.read());r=run(d['command'],d.get('payload',''));print(r.stdout,end='');print(r.stderr,file=sys.stderr,end='');sys.exit(r.returncode)
