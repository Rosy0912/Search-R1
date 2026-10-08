import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='4'
import pathlib,importlib.util,json,tempfile,datetime
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
def child(rank,rendezvous,directory):
 torch.set_num_threads(1)
 import search_r1.baselines.training_state
 p=pathlib.Path('work/tree-ppo-three-guards-review/sampling-interface/reviewed-snapshot/search_r1/baselines/sampling_consistency.py')
 spec=importlib.util.spec_from_file_location('search_r1.baselines.review_sampling',p);M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
 dist.init_process_group('gloo',init_method='file://'+rendezvous,world_size=2,rank=rank,timeout=datetime.timedelta(seconds=3))
 result={}
 try:
  actor=torch.zeros(2,3);sampled=torch.zeros(2,3 if rank==0 else 2)
  neutral=dict(temperature=1.,top_p=1.,top_k=-1,min_p=0.,presence_penalty=0.,frequency_penalty=0.,repetition_penalty=1.,n=1,logits_processors=[])
  M.reduce_pair_statistics(sampled,actor,torch.ones_like(actor),neutral,1.,.01)
  result={'unexpected_accepted':True}
 except Exception as e:result={'error_type':type(e).__name__,'error':str(e)}
 finally:
  pathlib.Path(directory,f'rank-{rank}.json').write_text(json.dumps(result));dist.destroy_process_group()
if __name__=='__main__':
 with tempfile.TemporaryDirectory() as directory:
  mp.spawn(child,args=(directory+'/rendezvous',directory),nprocs=2,join=True)
  results=[json.loads(pathlib.Path(directory,f'rank-{i}.json').read_text()) for i in range(2)]
 assert 'size of tensor' in results[1]['error'] and ('peer' in results[0]['error'].lower() or 'closed' in results[0]['error'].lower() or 'timed' in results[0]['error'].lower()),results
 pathlib.Path('work/tree-ppo-three-guards-review/sampling-interface/GLOO_PREPARE_COUNTEREXAMPLE.json').write_text(json.dumps({'actual_two_cpu_gloo_ranks':True,'synchronized_prepare_refusal':False,'results':results},indent=2)+'\n');print(results)
