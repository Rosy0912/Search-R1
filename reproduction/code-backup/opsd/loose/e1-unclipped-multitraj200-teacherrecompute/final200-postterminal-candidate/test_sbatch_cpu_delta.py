"""Local argv/environment delta only, never invoke Slurm or production main."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
OLD=HERE/'final200_postterminal_cpu.py.v1-reviewed-62b05d1f8527.original'
OLD_SHA='62b05d1f8527cb0c772741ac8adadf103b4ad3f8a6a8a7527e0d665e6df056bf'
OLD_RAW=OLD.read_bytes()
assert hashlib.sha256(OLD_RAW).hexdigest()==OLD_SHA
NEW_RAW=(HERE/'final200_postterminal_cpu.py').read_bytes()

def load(raw,path,name):
 m=types.ModuleType(name);m.__file__=str(path)
 exec(compile(raw,str(path),'exec'),m.__dict__);return m

old=load(OLD_RAW,OLD,'sealed_postterminal_v1_cpu_delta')
new=load(NEW_RAW,HERE/'final200_postterminal_cpu.py','candidate_postterminal_v2_cpu_delta')

class SubmissionDelta(unittest.TestCase):
 def test_only_sbatch_function_changes_in_runtime(self):
  before=b'def sbatch_command(script):\n';after=b'def srun_command(job,intent,sha):\n'
  self.assertEqual(OLD_RAW.split(before,1)[0],NEW_RAW.split(before,1)[0])
  self.assertEqual(OLD_RAW.split(after,1)[1],NEW_RAW.split(after,1)[1])
  self.assertEqual(new.SOURCE,old.SOURCE);self.assertEqual(new.BUNDLE_SHA,old.BUNDLE_SHA)
  argv=new.sbatch_command(new.OUTPUT/'batch.original.sh')
  self.assertEqual(argv[0],'env');self.assertIn('--mem=128G',argv);self.assertIn('--nodelist=h100-10',argv)
  self.assertFalse(any(v.startswith(('--gres','--gpus','--exclusive')) for v in argv))
 def test_real_local_env_removes_inherited_submission_options(self):
  poison={'SBATCH_GRES':'gpu:h100:8','SBATCH_GPUS':'8','SBATCH_EXCLUSIVE':'yes',
          'SBATCH_CLUSTERS':'foreign-fixture','SRUN_GPUS':'8','SLURM_CLUSTERS':'foreign-fixture',
          'SLURM_JOB_ID':'3304','SLURM_CONF':'/not-used/fixture.conf',
          'POSTTERMINAL_CPU_FIXTURE_KEEP':'retained-local-value'}
  with mock.patch.dict(os.environ,poison,clear=False):
   argv=new.sbatch_command(new.OUTPUT/'batch.original.sh')
   at=argv.index('sbatch')
   # Substitute the command with this local interpreter; no Slurm command runs.
   script='import json,os,sys;keys=json.loads(sys.argv[1]);print(json.dumps({k:os.environ.get(k) for k in keys},sort_keys=True))'
   call=argv[:at]+[sys.executable,'-c',script,json.dumps(list(poison))]
   result=subprocess.run(call,capture_output=True,check=False)
  self.assertEqual(result.returncode,0,result.stderr.decode())
  observed=json.loads(result.stdout)
  for key in poison:
   if key.startswith(('SBATCH_','SRUN_','SLURM_')):self.assertIsNone(observed[key])
  self.assertEqual(observed['POSTTERMINAL_CPU_FIXTURE_KEEP'],'retained-local-value')
 def test_cpu_step_command_and_batch_script_unchanged(self):
  intent=new.OUTPUT/'intent.original.json';sha='a'*64
  with mock.patch.object(old,'SELF',new.SELF):
   self.assertEqual(new.srun_command('901',intent,sha),old.srun_command('901',intent,sha))
  self.assertIn('--gres=none',new.srun_command('901',intent,sha))
  # The unchanged batch script embeds its own source filename, so compare
  # its function bytecode and constants rather than pretending both paths match.
  self.assertEqual(new.batch_script.__code__.co_code,old.batch_script.__code__.co_code)
  self.assertEqual(new.batch_script.__code__.co_consts,old.batch_script.__code__.co_consts)

if __name__=='__main__':unittest.main(verbosity=2)
