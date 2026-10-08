import pathlib,json,time,hashlib,subprocess
R=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-safety300-3268');o={'epoch':time.time(),'files':{},'protected':{}}
for n in ['resource-samples.jsonl','process-cost.jsonl','cost.json','run-result.json','launch-manifest.json','resolved.yaml','output/entry-audit/entry.jsonl','output/entry-audit/entry-config.yaml','retriever-coexistence.json','stage.log']:
 p=R/n
 if p.is_file():b=p.read_bytes();o['files'][n]={'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode()}
for pid in (1680782,1680783,1680784,1680785,1696443,1696444,1696445,1696446):o['protected'][str(pid)]=pathlib.Path(f'/proc/{pid}/stat').read_text()
o['gpu_now']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
o['gpu_processes_now']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
o['ports']=subprocess.run(['ss','-ltnp'],text=True,capture_output=True).stdout
o['output_file_counts']={name:len(list((R/'output').glob(pat))) for name,pat in [('direct_rank','tree-rank-audit/step-*.json'),('ratio','tree-rank-audit/ratio-tail/*'),('checkpoints','checkpoints/**/*'),('trajectory_manifests','trajectories/outer-*/manifest.json'),('tree_audits','tree-audit/*.json'),('sampling_json','sampling-consistency/*.json'),('monitor_pt','probability-calibration/*-monitor.pt')]}
print(json.dumps(o))
