"""CPU-only preparation of immutable eight-GPU service. No GPU activation."""
import argparse,hashlib,json,os,secrets,shlex,socket,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--base',required=True);a=p.parse_args()
source=Path(a.source).resolve();b=Path(a.base).resolve()
assert os.getuid()==1005 and socket.gethostname().split('.')[0]=='xielab1'
assert source.parent==Path('/data/shaoshitong/recursive-bran-a-8gpu-20261011')
assert not b.exists(),'CPU service preparation is one-shot'
mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines());assert int(mem['MemAvailable'].split()[0])*1024>=200*1024**3
assert ':18006' not in subprocess.check_output(['ss','-ltn'],text=True)
oracle=Path('/proc/2239529');assert oracle.stat().st_uid==1005 and int((oracle/'stat').read_text().rsplit(') ',1)[1].split()[19])==914664099
b.mkdir();(b/'receipts').mkdir();(b/'logs').mkdir()
token=b/'receipts/private-probe-token';token.write_text(secrets.token_hex(32));token.chmod(0o600)
helper=source/'release/recursive_bran_a/retriever_overlap_candidate'
old=Path('/data/shaoshitong/search-e1-retriever-segmented-20261007')
python=str(old/'env/bin/python');runtime='/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/runtime/opsd/upstream/Search-R1'
env=dict(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',PYTHONPATH=str(helper)+':'+str(old/'source')+':'+runtime,
 PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='16',TOKENIZERS_PARALLELISM='false')
command=''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items())
command+='export E1_SEGMENTED_PROBE_TOKEN="$(cat '+shlex.quote(str(token))+')"\n'
command+='exec '+shlex.join([python,'-B','-u',str(helper/'eight_gpu_server.py'),'--oracle-source',str(old/'source'),
 '--base',str(b),'--assets',str(old/'assets'),'--asset-manifest',str(old/'assets/manifest.json'),'--port','18006'])
command+=' >'+shlex.quote(str(b/'logs/server.log'))+' 2>&1\n'
f=b/'receipts/cpu-launch.sh';f.write_text(command);f.chmod(0o700)
session='rba-eight-retriever-20261012';subprocess.run(['tmux','new-session','-d','-s',session,'bash '+shlex.quote(str(f))],check=True)
subprocess.run(['tmux','set-option','-t',session,'remain-on-exit','on'],check=True)
proof=dict(time=time.time(),source=str(source),source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
 base=str(b),session=session,launch_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),gpu_activation=False)
(b/'receipts/cpu-start.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
