import sys,subprocess,shlex,json,time,uuid,base64,hashlib
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
prep=Path('/Users/shaoshitong/Documents/Search-G1/preparation');repo=prep.parent/'repo'
expected=json.loads((prep/'bootstrap-verified-source-manifest.json').read_text());retained=json.loads((prep/'bootstrap-retained-source-proof.json').read_text())['files']
selected={n:h for n,h in expected.items() if n.startswith('wheelhouse/') and any(Path(n).name.startswith(s+'-') for s in ('accelerate','pybind11','matplotlib','contourpy','cycler','fonttools','kiwisolver','pyparsing'))}
sizes={n:retained[n]['size'] for n in selected}
export='/dataset/shaoshitong/g1-common8-repair-wheels-'+uuid.uuid4().hex
source=(repo/'reproduction/common8/assets_export.py').read_text()
start=source.index('FILES=');end=source.index('\nSNAPSHOT=',start)
source=source[:start]+'FILES='+repr(sizes)+source[end:]
source=source.replace("BASE=Path('/dataset/shaoshitong/g1-common8-original-retrieval-export')",'BASE=Path('+repr(export)+')').replace('19874','19875')
code="import base64,json,os,subprocess,time;from pathlib import Path\nbase=Path("+repr(export)+");base.mkdir(exist_ok=False);script=base/'server.py';script.write_bytes(base64.b64decode("+repr(base64.b64encode(source.encode()).decode())+"))\n"+r"""env=dict(os.environ,CUDA_VISIBLE_DEVICES='');env.pop('LD_LIBRARY_PATH',None)
with (base/'server.log').open('x') as f:p=subprocess.Popen(['nice','-n','19','python3',str(script)],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(100):
 if (base/'handle.json').exists():break
 assert p.poll() is None;time.sleep(.1)
h=json.loads((base/'handle.json').read_text());assert h['pid']==p.pid;print(json.dumps(h))
"""
r=subprocess.run(control.route('h100-9','python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=55);assert r.returncode==0,r.stderr
handle=json.loads(r.stdout);(prep/'repair-wheels-export-handle.json').write_text(json.dumps(dict(export=export,**handle),indent=2)+'\n')
try:
 code="import json,hashlib,urllib.request,subprocess,sys;from pathlib import Path\nexpected=json.loads("+repr(json.dumps(selected))+")\n"+r"""base=Path('/dataset/shaoshitong/Search-G1-common8');target=base/'bootstrap-assets/repair-deps';target.mkdir(exist_ok=True)
opened=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for n,h in expected.items():
 path=target/Path(n).name
 if not path.exists():
  with opened.open('http://10.33.0.13:19875/'+n,timeout=30) as response,path.open('xb') as f:f.write(response.read())
 assert hashlib.sha256(path.read_bytes()).hexdigest()==h
 subprocess.run(['/root/miniconda3/bin/python','-m','pip','--python',str(base/'env/bin/python'),'install','--no-deps',str(path)],check=True,stdout=sys.stderr,stderr=sys.stderr)
from accelerate import init_empty_weights
import torch
with init_empty_weights():model=torch.nn.Linear(4,4)
assert model.weight.device.type=='meta'
import pybind11,matplotlib
print(json.dumps(dict(accepted=True,files_sha256=expected,actual_empty_weights_context=True,versions={'accelerate':__import__('accelerate').__version__,'pybind11':pybind11.__version__,'matplotlib':matplotlib.__version__},scope='actual target CPU imports and meta initializer, same-cluster source wheels; no GPU/model/GRPO')))
"""
 compile(code,'wheel-client','exec')
 r=subprocess.run(control.route('h100-8','/dataset/shaoshitong/Search-G1-common8/env/bin/python -c '+shlex.quote(code)),capture_output=True,text=True,timeout=55)
 proof=dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,checked_unix=time.time());(prep/'repair-deps-same-cluster-observation.json').write_text(json.dumps(proof,indent=2)+'\n');print(r.stdout,r.stderr)
finally:
 code="import json,os,signal,time;from pathlib import Path\nh=json.loads("+repr(json.dumps(handle))+")\n"+r"""p=Path('/proc')/str(h['pid']);stat=(p/'stat').read_text().rsplit(')',1)[1].split();assert (int(stat[19]),int(stat[3]),p.stat().st_uid)==(h['start_ticks'],h['sid'],h['uid'])
fd=os.pidfd_open(h['pid']);signal.pidfd_send_signal(fd,signal.SIGTERM);os.close(fd)
for _ in range(50):
 if not p.exists():break
 if p.readlink() if p.is_symlink() else False:raise AssertionError('Unexpected proc symlink')
 if (p/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z':break
 time.sleep(.1)
assert not p.exists() or (p/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z';print(json.dumps(dict(cleaned=True,handle=h)))
"""
 r=subprocess.run(control.route('h100-9','python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=55);assert r.returncode==0,r.stderr
 (prep/'repair-wheels-source-cleanup.json').write_text(r.stdout)
