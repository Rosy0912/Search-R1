import pathlib,sys,ast,types,importlib.util,tempfile,json,torch
from pytest import MonkeyPatch
R=pathlib.Path(__file__).resolve().parents[2];C=R/'work/tree-numeric-candidate';sys.path.insert(0,str(C))
source=C/'tests/baselines/test_tree_numeric_failure.py';spec=importlib.util.spec_from_file_location('tree_test',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from search_r1.baselines.tree_rank_audit import TreeRankAudit
from verl.workers.actor.dp_actor import DataParallelPPOActor
with tempfile.TemporaryDirectory() as tmp:
 root=pathlib.Path(tmp);mp=MonkeyPatch();sentinel=ValueError('Nonfinite independent sentinel');captured=[]
 try:
  mp.setattr(DataParallelPPOActor,'_optimizer_step',lambda self:(_ for _ in ()).throw(sentinel))
  actual=TreeRankAudit.failure
  def observe(self,error):captured.append(error);actual(self,error)
  mp.setattr(TreeRankAudit,'failure',observe)
  file=root/'nondirectory';file.write_text('no');m.run_actor(mp,file,True,corrupt=True);assert captured==[sentinel]
 finally:mp.undo()
 # Prove intentional forward/KL boundary absentfailure capture using real actor.
 text=source.read_text().replace('use_kl_loss=False','use_kl_loss=True').replace("old_log_probs=torch.zeros(4, 3)","ref_log_prob=torch.zeros(4, 3), old_log_probs=torch.zeros(4, 3)").replace('return logp * 0, logp','return logp * 0, logp * torch.tensor(float("nan"))')
 namespace=dict(m.__dict__);exec(compile(text,str(source),'exec'),namespace);mp=MonkeyPatch()
 try:
  namespace['run_actor'](mp,root/'kl',True,corrupt=True);assert not list((root/'kl').rglob('*.pt'))
 finally:mp.undo()
out=dict(status='SCOPED_BOUNDARIES_CONFIRMED',TreeDiagnosticDiskFailure_preserves_same_optimizer_exception_object=True,forward_KL_construct_error_no_failure_tensor_capture=True,scope='Exact current source real CPU actor; latter intentionally excluded by narrowed recovery plan')
(R/'work/tree-numeric-candidate-review/boundaries.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
