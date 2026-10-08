"""Source identity and actual delivered teacher, without extra pretraining proofs."""
import json,pathlib,hashlib
from pipeline import verify_files
from checkpoint_inventory import sha
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
def check(c,read_native=False):
 if c.get('mode')!='topd300' or c.get('expected_outer_steps')!=300:raise ValueError('Only actual student300')
 for name in ('source_manifest','tools_manifest','launcher_manifest','teacher_backend_manifest'):
  ref=c[name];p=pathlib.Path(ref['path']);base=p.parent
  if ROOT not in p.resolve().parents or sha(p)!=ref['sha256']:raise ValueError('Owned source bytes differ '+name)
  record=json.loads(p.read_text());files=dict(record['files']);files[p.name]={'size':p.stat().st_size,'sha256':ref['sha256']};verify_files(base,files)
 if c['base_model']!=str(ROOT/'assets/qwen2.5-3b-instruct-aa8e725'):raise ValueError('Official Instruct student path differs')
 if c['train_file']!=str(ROOT/'assets/common-instruct-0b002b/train.parquet') or c['val_file']!=str(ROOT/'assets/common-instruct-0b002b/validation.parquet'):raise ValueError('Shared common data path differs')
 if not c.get('teacher_actual_result') or not c.get('teacher_path'):raise ValueError('Actual GRPO100 teacher is pending')
 ref=c['teacher_actual_result']
 if sha(ref['path'])!=ref['sha256']:raise ValueError('Actual teacher result identity differs')
 result=json.loads(pathlib.Path(ref['path']).read_text())
 if result['status']!='COMPLETE_UNREVIEWED' or not result['cleanup_passed']:raise ValueError('Teacher100 not delivered')
 outer=result['actual_outer_evidence']
 if sha(outer['path'])!=outer['sha256']:raise ValueError('Actual teacher outer receipt drift')
 actual=json.loads(pathlib.Path(outer['path']).read_text())
 if actual['actual_completed_outer_steps']!=list(range(1,101)) or actual['export_step']!=100:raise ValueError('Need actual100, not profile or2+98')
 if pathlib.Path(result['export']['directory']).resolve()!=pathlib.Path(c['teacher_path']).resolve():raise ValueError('Teacher full HF export path differs')
 return result
