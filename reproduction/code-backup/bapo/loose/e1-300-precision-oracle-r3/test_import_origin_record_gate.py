"""Actual local origin maps plus polluted-path/hash/missing-record refusals."""
import ast,importlib.util,json,copy,tempfile,unittest,hashlib
from pathlib import Path
base=Path(__file__).parent;tree=ast.parse((base/'tools/verify_launch.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='validate_import_origins');ns=dict(pathlib=__import__('pathlib'),sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest());exec(compile(ast.Module(body=[fn],type_ignores=[]),'origin-gate','exec'),ns);gate=ns['validate_import_origins']
class OriginGate(unittest.TestCase):
 def actual(self):
  repo=(base.parent/'bapo-e1-300-devicefix').resolve();log=(base.parent/'evidence/e1-300-devicefix-r2-source-functions-local-tests-stable-final.log').read_text();proof={}
  for key,marker in [('before','CPU_IMPORT_ORIGINS_BEFORE_JSON='),('after','CPU_IMPORT_ORIGINS_AFTER_JSON=')]:proof[key]=json.loads(next(line.split(marker,1)[1]for line in log.splitlines()if marker in line))
  return repo,proof
 def test_actual_fullsource_before_after_origin_records_pass(self):
  repo,p=self.actual();gate(repo,'733097feeae371b4efd8fd48940c46ff36da839b',p)
 def test_old_missing_hash_and_identity_records_refused(self):
  repo,p=self.actual()
  for kind in ('old','missing','hash','source','repo','beforeafter','functionold','functionmissing','functionchanged'):
   bad=copy.deepcopy(p)
   if kind=='old':bad['after']['modules']['verl.workers.actor.dp_actor']['path']='/dataset/shaoshitong/search-r1-rl/bapo/repo/verl/workers/actor/dp_actor.py'
   if kind=='missing':bad['after']['modules'].pop('search_r1.baselines.full_state')
   if kind=='hash':bad['after']['modules']['verl']['sha256']='0'*64
   if kind=='source':bad['after']['source_commit']='f0e37bf68d61ec926ac225f5e1ee1f606cb09bf9'
   if kind=='repo':bad['before']['repo']='/old/repo'
   if kind=='beforeafter':bad['after']['modules'].pop('verl.utils.logging_utils')
   name='verl.workers.actor.dp_actor.DataParallelPPOActor.update_policy'
   if kind=='functionold':bad['after']['functions'][name]['co_filename']='/old/repo/verl/workers/actor/dp_actor.py'
   if kind=='functionmissing':bad['after']['functions'].pop(name)
   if kind=='functionchanged':bad['after']['functions'][name]['code_sha256']='0'*64
   with self.assertRaises((AssertionError,FileNotFoundError),msg=kind):gate(repo,'733097feeae371b4efd8fd48940c46ff36da839b',bad)
 def test_stage_only_newrepo_and_runtime_proof_required(self):
  s=(base/'tools/stage.sh').read_text();self.assertIn('export PYTHONPATH="$ROOT/repo"',s);self.assertIn('cpu-tests.py" --origins-only',s)
  self.assertIn("'runtime-import-origins.log'",(base/'tools/verify_hardware_receipt.py').read_text());self.assertIn("'runtime-import-origins.log'",(base/'capture-hardware-devicefix.py').read_text())
if __name__=='__main__':unittest.main()
