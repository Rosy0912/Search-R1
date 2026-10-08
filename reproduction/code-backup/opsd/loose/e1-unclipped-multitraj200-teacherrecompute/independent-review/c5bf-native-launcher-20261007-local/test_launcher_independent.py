import ast, contextlib, copy, hashlib, io, json, os, signal, subprocess, sys, tempfile, types, unittest
from pathlib import Path
from unittest import mock
SOURCE=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-4/work/e1-unclipped-multitraj200-teacherrecompute/c5bf_native_cpu_launch_once_candidate.py')
RAW=SOURCE.read_bytes()
EXPECTED='7731a24dcadf631b899003ba66b8b7f2cfb0e7fcc39b34e1cfab8869174b0196'
assert hashlib.sha256(RAW).hexdigest()==EXPECTED
M=types.ModuleType('launcher_review_fixture'); M.__file__=str(SOURCE)
exec(compile(RAW,str(SOURCE),'exec',dont_inherit=True,optimize=0),M.__dict__)
TREE=ast.parse(RAW)
MAIN=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=='main')
def extracted_function(nodes,name,returns):
 f=ast.FunctionDef(name=name,args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=copy.deepcopy(nodes)+[ast.Return(value=ast.parse(returns,mode='eval').body)],decorator_list=[])
 t=ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[]))
 exec(compile(t,'EXTRACTED_LOCAL_FIXTURE_ONLY','exec'),M.__dict__)
 return getattr(M,name)
start=next(i for i,n in enumerate(MAIN.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='environment' for t in n.targets))
end=next(i for i,n in enumerate(MAIN.body[start:],start) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='write_once')
ENV=extracted_function(MAIN.body[start:end],'only_environment_fixture','(environment,removed)')
start=next(i for i,n in enumerate(MAIN.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='sampling_error' for t in n.targets))
end=next(i for i,n in enumerate(MAIN.body[start:],start) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='after' for t in n.targets))
WAIT=extracted_function(MAIN.body[start:end],'only_wait_fixture','(code,child.pid,sampling_error)')
class Review(unittest.TestCase):
 def test_command_scope_and_exact_dependencies(self):
  a=M.command()
  self.assertEqual(a[:8],['srun','--jobid=3304','--overlap','--nodes=1','--ntasks=1','--cpus-per-task=4','--gres=none','--nodelist=h100-10'])
  self.assertIn('CUDA_VISIBLE_DEVICES=',a);self.assertIn('-B',a)
  self.assertNotIn('sbatch',a);self.assertNotIn('--exclusive',a)
  self.assertEqual(a[a.index('--source-inventory-sha256')+1],M.INVENTORY_SHA)
  self.assertEqual(a[a.index('--self-sha256')+1],M.CONTROLLER_SHA)
  self.assertEqual(a[a.index('--checker-sha256')+1],M.CHECKER_SHA)
  self.assertEqual(a[a.index('-c')+1],M.CPU_BOOTSTRAP)
  self.assertFalse(any(x.startswith(('--gpus','--model','--stage','--eval')) for x in a))
 def test_polluted_environment_removed(self):
  e={'SBATCH_GRES':'gpu:8','SRUN_CPUS_PER_TASK':'9','SLURM_JOB_ID':'fake','OPSD_FINAL200_X':'fake','OPSD_E1_RETRIEVER_URL':'remote','KEEP':'yes','CUDA_VISIBLE_DEVICES':'0'}
  with mock.patch.dict(os.environ,e,clear=True):r,n=ENV()
  self.assertEqual(r,{'KEEP':'yes','CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
  self.assertEqual(set(n),set(e)-{'KEEP','CUDA_VISIBLE_DEVICES'})
 def test_bootstrap_identity_and_cuda_rejected_before_source(self):
  cases=[{'SLURM_JOB_ID':'33040','SLURM_STEP_ID':'2','CUDA_VISIBLE_DEVICES':''},{'SLURM_JOB_ID':'3304','SLURM_STEP_ID':'../2','CUDA_VISIBLE_DEVICES':''},{'SLURM_JOB_ID':'3304','SLURM_STEP_ID':'2','CUDA_VISIBLE_DEVICES':'0'}]
  for e in cases:
   with self.subTest(e=e),mock.patch.dict(os.environ,e,clear=True),mock.patch.object(sys,'argv',['-c','/file-must-not-be-read','bad']):
    with self.assertRaises(ValueError):exec(compile(M.CPU_BOOTSTRAP,'LOCAL_BOOTSTRAP_REJECTION_FIXTURE','exec'),{'__builtins__':__builtins__})
 def test_bootstrap_same_bytes_and_hash_guard(self):
  # UID0 is a declared synthetic metadata fixture; real file/open/hash/compile bytes remain actual local bytes.
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp).resolve()/'fixture.py';p.write_text('import json,sys\nprint(json.dumps({"argv":sys.argv,"file":__file__}))\n')
   raw=p.read_bytes();lstat=Path.lstat;fstat=os.fstat
   def fake(s):
    return types.SimpleNamespace(**{k:(0 if k=='st_uid' else getattr(s,k)) for k in ('st_dev','st_ino','st_size','st_uid','st_mode','st_nlink','st_mtime_ns','st_ctime_ns')})
   def ls(q,*args,**kw):return fake(lstat(q,*args,**kw))
   def fs(fd):return fake(fstat(fd))
   e={'SLURM_JOB_ID':'3304','SLURM_STEP_ID':'17','CUDA_VISIBLE_DEVICES':''}
   for expected,good in [(hashlib.sha256(raw).hexdigest(),True),('0'*64,False)]:
    out=io.StringIO()
    with mock.patch.dict(os.environ,e,clear=True),mock.patch.object(sys,'argv',['-c',str(p),expected,'--fixture-only']),mock.patch.object(Path,'lstat',ls),mock.patch.object(os,'fstat',fs),contextlib.redirect_stdout(out):
     if good:exec(compile(M.CPU_BOOTSTRAP,'LOCAL_SAME_BYTES_FIXTURE','exec'),{'__builtins__':__builtins__})
     else:
      with self.assertRaisesRegex(ValueError,'execution SHA'):exec(compile(M.CPU_BOOTSTRAP,'LOCAL_BAD_HASH_FIXTURE','exec'),{'__builtins__':__builtins__})
    if good:
     self.assertEqual(json.loads(out.getvalue()),{'file':str(p),'argv':[str(p),'--fixture-only','--execution-job-id','3304','--execution-step-id','17']})
    else:self.assertEqual(out.getvalue(),'')
 def run_wait(self,code,metadata_error=False):
  tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
  base=Path(tmp.name).resolve();launch=base/'launch';launch.mkdir()
  # Exact wait/body AST, only command/globals are local fixtures. No main/context/srun/remote invocation.
  M.BASE=base;M.LAUNCH=launch;M.argv=[sys.executable,'-B','-c',code];M.environment=dict(os.environ)
  def process(pid):
   if metadata_error:raise ValueError('SYNTHETIC_LOCAL_METADATA_FAILURE')
   return {'pid':pid,'scope':'SYNTHETIC_LOCAL_METADATA_ONLY'}
  with mock.patch.object(M,'process',process):rc,pid,error=WAIT()
  r=json.loads((launch/'srun-wait.original.json').read_text())
  self.assertEqual(r['returncode'],rc);self.assertIs(r['waited'],True);self.assertEqual(r['pid'],pid)
  self.assertEqual(r['start_record_error'],error)
  self.assertEqual((launch/'srun.stdout.original').read_bytes(),b'local stdout\n')
  self.assertEqual((launch/'srun.stderr.original').read_bytes(),b'local stderr\n')
  self.assertNotIn(pid,subprocess._active)
  return rc,error
 def test_wait_real_child_zero(self):
  rc,e=self.run_wait('import sys; print("local stdout",flush=True);print("local stderr",file=sys.stderr,flush=True)')
  self.assertEqual(rc,0);self.assertIsNone(e)
 def test_wait_real_child_nonzero(self):
  rc,e=self.run_wait('import sys;print("local stdout",flush=True);print("local stderr",file=sys.stderr,flush=True);sys.exit(7)')
  self.assertEqual(rc,7);self.assertIsNone(e)
 def test_wait_real_child_signal(self):
  rc,e=self.run_wait('import os,signal,sys;print("local stdout",flush=True);print("local stderr",file=sys.stderr,flush=True);os.kill(os.getpid(),signal.SIGTERM)')
  self.assertEqual(rc,-15);self.assertIsNone(e)
 def test_metadata_failure_preserves_actual_wait(self):
  rc,e=self.run_wait('import sys;print("local stdout",flush=True);print("local stderr",file=sys.stderr,flush=True);sys.exit(6)',True)
  self.assertEqual(rc,6);self.assertEqual(e['type'],'ValueError')
if __name__=='__main__':unittest.main(verbosity=2)
