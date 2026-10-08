import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
import pathlib,hashlib,json,time,resource,sys,torch
P=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/safety110-a632ba9-84e3e186');R=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-safety300-3268');sys.path.insert(0,str(P/'source/repo'));torch.set_num_threads(4)
from search_r1.baselines.training_state import digest
start=time.monotonic();cpu=time.process_time();rows=[]
for rank in range(8):
 n=R/f'native-fullstate/rank-{rank:02d}.json';j=json.loads(n.read_bytes());d=json.loads((R/f'native-fullstate/rank-{rank:02d}-diagnostic.json').read_bytes());p=pathlib.Path(j['path']);b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==j['sha256'];v=torch.load(p,map_location='cpu',weights_only=False);assert digest(v)==j['logical_state_sha256'];del v
 assert j['identity']==dict(rank=rank,world_size=8,job='3268',code='a632ba91840fb116aacfc0e770681a860647b295')
 assert j['status']=='PASS' and j['full_worker_state_finite'] and j['exact_same_world_reload_verified'] and j['cold_new_wrapper_verified'] and j['same_world_sixth_outer_replay_equal'] and j['optimizer_steps_at5']==120 and j['optimizer_steps_at6']==130 and j['recovery_optimizer_calls']==10
 assert d['rank']==rank and d['differences']==[] and d['replayed_digest']==d['uninterrupted_digest']
 for mode,key in [('uninterrupted','uninterrupted_digest'),('replayed','replayed_digest')]:
  p=R/f'native-fullstate/rank-{rank:02d}-{mode}6.pt'
  assert p.is_file();v=torch.load(p,map_location='cpu',weights_only=False);assert digest(v)==d[key];del v
 k=json.loads((R/f'gpu-kl-proof/rank-{rank}.json').read_bytes());assert k['status']=='PASS' and k['rank']==rank and k['torch']=='2.4.0+cu121' and pathlib.Path(k['core_algos_file'])==P/'source/repo/verl/trainer/ppo/core_algos.py';assert {x['dtype'] for x in k['records']}=={'torch.float32','torch.bfloat16','torch.float16'}
 for x in k['records']:assert x['native_dtype_values_and_gradients_exact'] and x['finite_backward'] and x['nonfinite_inputs_refused'] and x['native_dtype_points_compared']==402
 rows.append(dict(rank=rank,native_json_sha256=hashlib.sha256(n.read_bytes()).hexdigest(),native_pt_sha256=j['sha256'],logical_digest=j['logical_state_sha256'],sixth_replay_digest=d['replayed_digest'],gpu_kl_records=3))
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='ACTUAL_NATIVE_SYNTHETIC_8RANK_AND_KL24_PASS_ONLY',rows=rows,scope='Tiny FSDP cold120to130 replay and synthetic CUDA KL; not actual3B zero-write calibration acceptance',actual_3b_calibration_accepted=False,wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,torch_threads=4,cuda_initialized=False)))
