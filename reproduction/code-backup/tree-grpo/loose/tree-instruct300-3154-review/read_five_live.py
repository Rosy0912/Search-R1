from pathlib import Path
import time,json,hashlib,subprocess
r=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-instruct300-3154');g=r/'output/acceptance-step5';resources=[json.loads(z) for z in (r/'resource-samples.jsonl').read_text().splitlines()];last=resources[-1];now=time.time()
assert last['job_id']=='3154' and not last.get('event') and len(last['gpus'])==8
assert all(not z.get('event') for z in resources)
foreign={z['pid'] for gpu in last['gpus'] for z in gpu['foreign']};assert foreign=={1680782,1680783,1680784,1680785,1696443,1696444,1696445,1696446}
assert all(gpu['foreign_memory_fraction']<=.15 and gpu['foreign_sm_upper']<=15 for gpu in last['gpus'])
s=subprocess.check_output(['scontrol','show','job','3154'],text=True);assert 'JobState=RUNNING' in s and 'Restarts=0' in s
assert not (g/'approved.json').exists() and not (g/'released.json').exists() and not list((r/'output/tree-rank-audit').glob('step-000006-rank-*.json'))
print(json.dumps({'status':'LIVE3154_WAITING_GATE_NO6_GUARD_BACKGROUND_PASS','epoch':now,'slurm':s,'last_guard':last,'guard_samples':len(resources),'ready_sha256':hashlib.sha256((g/'ready.json').read_bytes()).hexdigest(),'whole_gpu_sampled_peak_bytes':max(gpu['total_used_memory_bytes'] for z in resources for gpu in z['gpus'])}))
