import os,json,subprocess,pathlib,datetime,fcntl
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'uid':os.getuid(),'hostname':subprocess.check_output(['hostname'],text=True).strip()}
for name,args in {'scheduler':['squeue','-h','-w','h100-10','-o','%i %u %t %j %N %M'],'gpu':['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],'apps':['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],'listeners':['ss','-lntp']}.items():
 r=subprocess.run(args,capture_output=True,text=True,timeout=20);out[name]={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
for path in ['/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py','/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.11/site-packages/transformers/models/qwen2/modeling_qwen2.py']:
 p=pathlib.Path(path)
 if p.is_file():
  text=p.read_text(); i=text.index('class Qwen2ForCausalLM');out['qwen2_source_tail']=text[i:i+20000];out['qwen2_path']=path
print(json.dumps(out))
