"""Real 3B inference only; no teacher, RL updates, or training acceptance."""
import argparse,json,os,time,subprocess
from pathlib import Path
from vllm import LLM,SamplingParams
p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--rank',type=int,required=True);a=p.parse_args()
d=Path(a.directory);root=Path(os.environ['AGENT_ROOT']);t=time.perf_counter();cfg=json.loads((d/'rollout-config.json').read_text())
llm=LLM(model=str(root/'assets-root/assets/qwen2.5-3b'),tensor_parallel_size=1,
        gpu_memory_utilization=cfg['gpu_memory_utilization'],max_model_len=cfg['max_model_len'],enforce_eager=cfg['enforce_eager'],dtype=cfg['dtype'])
expected_uuid=os.environ['PROBE_EXPECTED_GPU_UUID']
visible=os.environ['CUDA_VISIBLE_DEVICES'];assert visible.isdecimal(), 'Pinned vLLM requires numeric physical CUDA selector'
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
actual_uuids={line.split(',')[0].strip() for line in apps.splitlines() if ',' in line and line.split(',')[1].strip()==str(os.getpid())}
assert actual_uuids=={expected_uuid},f'Own vLLM PID used {actual_uuids}, expected leased {expected_uuid}'
(d/f'ready-{a.rank}.json').write_text(json.dumps(dict(pid=os.getpid(),rank=a.rank,
    cuda_visible_devices=visible,expected_gpu_uuid=expected_uuid,actual_pid_gpu_uuids=sorted(actual_uuids),initialization_seconds=time.perf_counter()-t,
    tp=1,gpu_memory_utilization=cfg['gpu_memory_utilization'],max_model_len=cfg['max_model_len'],enforce_eager=cfg['enforce_eager'],dtype=cfg['dtype'],max_num_seqs="vllm0.6.3 default (as common patched wrapper does not pass max_num_seqs)",NO_TRAINING=True)))
deadline=time.monotonic()+1200
while not (d/'GO').exists():
    if time.monotonic()>deadline:raise TimeoutError('Own inference barrier timeout')
    time.sleep(.2)
queries=json.loads((d/'queries.json').read_text())[a.rank::8]
t=time.perf_counter();out=llm.generate(['Question: '+q+'\nAnswer:' for q in queries],SamplingParams(temperature=0,max_tokens=16))
assert len(out)==len(queries) and all(v.outputs[0].token_ids for v in out)
(d/f'done-{a.rank}.json').write_text(json.dumps(dict(rank=a.rank,requests=len(out),
    generation_seconds=time.perf_counter()-t,output_tokens=sum(len(v.outputs[0].token_ids) for v in out),NO_TRAINING=True)))
# Retain KV allocation until root has captured co-residency and retrieval completion.
while not (d/'RELEASE').exists():
    if time.monotonic()>deadline:raise TimeoutError('Own inference release timeout')
    time.sleep(.2)
