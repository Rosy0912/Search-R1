import os,time,json,hashlib,resource,gc
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import torch
from pathlib import Path
torch.set_num_threads(4);start=time.perf_counter();cpu=time.process_time();root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-recovery300-3234/output');directory=root/'trajectories/outer-000112'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
mp=directory/'manifest.json';m=json.loads(mp.read_bytes());p=directory/'actor-inputs.pt';actual=sha(p)
assert actual==m['payloads']['actor-inputs.pt']
payload=torch.load(p,map_location='cpu',weights_only=False);t=payload['tensors'];meta=payload['non_tensor_batch'];width=t['responses'].shape[1];effective=t['attention_mask'][:,-width:]*t['loss_mask'];startrow=6*480+7*48+20*2;rows=list(range(startrow,startrow+2));mask=effective[rows].bool();adv=t['advantages'][rows];negative=(adv<0)&mask
rpath=root/'tree-rank-audit/ratio-tail/step-000112-rank-06-optimizer-07.json';r=json.loads(rpath.read_bytes());micro=r['microbatches'][20]
assert r['rank']==6 and r['optimizer_call']==8 and mask.sum().item()==micro['valid_tokens']==2199
print(json.dumps(dict(status='PASS_READONLY112_TARGET_MICRO_MAPPING',rank=6,optimizer_call=8,microbatch_one_based=21,rows=rows,valid_tokens=int(mask.sum()),negative_advantage_tokens=int(negative.sum()),negative_advantage_values=sorted(set(adv[negative].tolist())),row_uid=[str(meta['uid'][i]) for i in rows],row_trajectory_uid=[str(meta['trajectory_uid'][i]) for i in rows],effective_per_row=mask.sum(-1).tolist(),negative_per_row=negative.sum(-1).tolist(),policy_mass=t['policy_weights'][rows].sum(-1).tolist(),raw_payload_sha256=actual,manifest_sha256=sha(mp),ratio_file_sha256=sha(rpath),wall_seconds=time.perf_counter()-start,process_cpu_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,torch_threads=4,cuda_initialized=torch.cuda.is_initialized(),scope='Readonly saved inputs select deterministic contiguous rank/minibatch/micro slice; actual valid-token count matches target ratio record. No saved full new_log_prob tensor, so cannot independently locate the peak token inside these two rows.')),flush=True)
