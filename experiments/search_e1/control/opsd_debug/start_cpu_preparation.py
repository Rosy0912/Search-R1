"""Start source-bound CPU case selection and weight hashing, no GPU activity."""
import hashlib,json,os,subprocess,time
from pathlib import Path
BASE=Path('/data/shaoshitong/search-e1-opsd-debug-20261008')
ORIGINAL=Path('/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007')
def main():
 assert os.getuid()==1005
 receipt=BASE/'receipts/cpu-preparation-start.json';assert not receipt.exists()
 sources=json.loads((BASE/'receipts/cpu-source-manifest.json').read_text())
 for name,h in sources.items():assert hashlib.sha256((BASE/'source/opsd_debug'/name).read_bytes()).hexdigest()==h
 assert not (BASE/'cases').exists() and not (BASE/'receipts/weight-manifest.json').exists()
 python=str(ORIGINAL/'env/bin/python');source=BASE/'source/opsd_debug'
 commands={'cases':[python,str(source/'select_cases.py'),'--runtime',str(ORIGINAL/'runtime/opsd-preparation3-xformers'),'--rollouts',str(ORIGINAL/'runs/preparation-3-rollouts.jsonl'),'--pairs',str(ORIGINAL/'runs/preparation-3-pairs.jsonl'),'--train',str(ORIGINAL/'data/train.jsonl'),'--output',str(BASE/'cases')], 'weights':[python,str(source/'prepare_weight_manifest.py'),'--base',str(ORIGINAL),'--output',str(BASE/'receipts/weight-manifest.json')]}
 env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']='';env['PYTHONUNBUFFERED']='1';children={}
 for name,argv in commands.items():
  logfile=BASE/'logs'/('cpu-'+name+'.log');handle=logfile.open('x')
  child=subprocess.Popen(argv,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True);handle.close()
  tick=int((Path('/proc')/str(child.pid)/'stat').read_text().rsplit(') ',1)[1].split()[19])
  children[name]=dict(pid=child.pid,uid=os.getuid(),starttick=tick,argv=argv,log=str(logfile))
 with receipt.open('x') as f:json.dump(dict(started_unix=time.time(),source_commit='7f029fb7860665a99cc94eb7e216b5b57c51e5c4',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cuda_visible_devices='',children=children,scope='CPU only; source datasets and weights unchanged; not live GPU diagnostic acceptance'),f,indent=2)
 print(json.dumps(dict(receipt=str(receipt),children=children)))
if __name__=='__main__':main()
