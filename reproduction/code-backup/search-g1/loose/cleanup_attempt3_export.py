import sys,json,subprocess,shlex,time
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
prep=Path('/Users/shaoshitong/Documents/Search-G1/preparation');observ=json.loads((prep/'attempt3-latest-observation.json').read_text());assert 'JobState=PENDING' in observ['job_stdout'] and 'DependencyNeverSatisfied' in observ['job_stdout'];assert 'run-manifest.json' not in observ;diag=json.loads((prep/'device-fix-native-observation.json').read_text());assert 'JobState=FAILED' in diag['job_stdout'];h=json.loads((prep/'attempt3-retrieval-export-handle.json').read_text())
code='import json,os,signal,time,hashlib;from pathlib import Path\nh='+repr(h)+'\n'+r'''
b=Path(h['export']);s=b/'serve_original_retrieval.py';assert hashlib.sha256(s.read_bytes()).hexdigest()==h['source_sha256']
p=Path('/proc')/str(h['pid'])
if p.exists():
 st=(p/'stat').read_text().rsplit(')',1)[1].split();assert (int(st[19]),int(st[3]),p.stat().st_uid)==(h['start_ticks'],h['sid'],h['uid']);assert str(s).encode() in (p/'cmdline').read_bytes()
 fd=os.pidfd_open(h['pid']);signal.pidfd_send_signal(fd,signal.SIGTERM);os.close(fd)
 for _ in range(50):
  if not p.exists() or (p/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z':break
  time.sleep(.1)
 assert not p.exists() or (p/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z'
proof=dict(cleaned=True,handle=h,checked_unix=time.time(),reason='native3244 numerical parity FAILED; dependent3245 never started; no staging needed')
(b/'cleanup.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
'''
r=subprocess.run(control.route('h100-9','python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=55);assert r.returncode==0,r.stderr
(prep/'attempt3-retrieval-source-cleanup.json').write_text(r.stdout);print(r.stdout)
