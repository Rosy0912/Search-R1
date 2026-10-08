import ast,datetime,pathlib,tempfile,types,unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from search_g1.numerics import require_finite


def rank_check(rank,store,queue):
 dist.init_process_group('gloo',init_method='file://'+store,rank=rank,world_size=2,timeout=datetime.timedelta(seconds=20))
 try:
  try:
   require_finite({'loss':torch.tensor(float('nan') if rank==1 else 1.)},stage='collective_test')
  except RuntimeError:queue.put((rank,'rejected'))
  else:queue.put((rank,'accepted'))
  try:require_finite({} if rank==1 else {'loss':torch.tensor(1.)},stage='empty_rank_test')
  except RuntimeError:queue.put((rank,'empty_rejected'))
  else:queue.put((rank,'empty_accepted'))
 finally:dist.destroy_process_group()

class NumericsGuard(unittest.TestCase):
 def test_finite_loss_preserved_and_empty_mask_rejected(self):
  require_finite({'loss':torch.tensor(2.),'count':torch.tensor(1)},stage='unit',positive=('count',))
  with self.assertRaises(RuntimeError):require_finite({'loss':torch.tensor(2.),'count':torch.tensor(0)},stage='unit',positive=('count',))
 def test_actual_optimizer_method_never_writes_nonfinite_gradient(self):
  source=pathlib.Path(__file__).resolve().parents[2]/'verl/workers/actor/dp_actor.py'
  tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and any(isinstance(x,ast.FunctionDef) and x.name=='_optimizer_step' for x in n.body));method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_optimizer_step')
  namespace=dict(torch=torch,FSDP=FSDP,require_finite=require_finite)
  exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
  model=torch.nn.Linear(2,1);opt=torch.optim.SGD(model.parameters(),lr=.1);initial=[p.clone() for p in model.parameters()]
  actor=types.SimpleNamespace(config=types.SimpleNamespace(grad_clip=1.),actor_module=model,actor_optimizer=opt)
  for p in model.parameters():p.grad=torch.full_like(p,float('nan'))
  with self.assertRaises(RuntimeError):namespace['_optimizer_step'](actor)
  self.assertTrue(all(torch.equal(a,p) for a,p in zip(initial,model.parameters())))
  for p in model.parameters():p.grad=torch.ones_like(p)
  namespace['_optimizer_step'](actor)
  self.assertTrue(any(not torch.equal(a,p) for a,p in zip(initial,model.parameters())))
 def test_other_rank_failure_rejects_all_ranks(self):
  context=mp.get_context('spawn');queue=context.Queue()
  with tempfile.TemporaryDirectory() as directory:
   processes=[context.Process(target=rank_check,args=(rank,str(pathlib.Path(directory)/'store'),queue)) for rank in range(2)]
   for p in processes:p.start()
   result=[queue.get(timeout=30) for _ in range(4)]
   for p in processes:p.join(timeout=10);self.assertEqual(p.exitcode,0)
   self.assertEqual(sorted(result),[(0,'empty_rejected'),(0,'rejected'),(1,'empty_rejected'),(1,'rejected')])
if __name__=='__main__':unittest.main()
