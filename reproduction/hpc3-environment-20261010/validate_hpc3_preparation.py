"""CPU-only checks of the prepared training configuration and immutable assets."""
import json
import os
import pathlib
import sys
import time

BASE = pathlib.Path('/data/user/jxu957/experiments/topd-20261008')
ROOT = BASE / 'runtime'
OUT = BASE / 'outputs/training-preparation'
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'tools/topd-instruct300-launch-v6/runtime'))
from checkpoint_inventory import inventory, sha
from student_binding import check
from supervise import verify_recipe
from official_student import verify
from retrieval_gate import verify_retrieval
from bind_hpc3_run import bind

def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2))
    print(json.dumps({'phase': name, 'epoch': time.time()}), flush=True)

c = json.loads((OUT / 'hpc3-config-draft.json').read_text())
verify_recipe(c)
approval = check(c)
bound = bind(c, os.environ['SLURM_JOB_ID'], os.environ['SLURMD_NODENAME'])
verify_recipe(bound)
assert bound['evidence_root'] != c['evidence_root']
assert 'baselines.local_runtime_host=' + os.environ['SLURMD_NODENAME'] in bound['training_overrides']
save('source-config-validation.json', {'status': 'EXACT_SOURCE_RECIPE_AND_TEACHER85_BINDINGS_PASS',
                                     'job': os.environ['SLURM_JOB_ID']})
student = inventory(c['base_model'], expected_role='student3b')
verify(student, c['student_model_identity'])
save('student-hf-fresh.json', student)
teacher = inventory(c['teacher_path'])
if approval['export'].get('files') is not None:
    assert teacher['files'] == approval['export']['files']
save('teacher-hf-fresh.json', teacher)
for ref in c['input_bindings']:
    assert sha(ref['path']) == ref['sha256']
import pyarrow.parquet as pq
rows = {pathlib.Path(r['path']).name: pq.read_metadata(r['path']).num_rows
        for r in c['input_bindings'] if r['path'].endswith('.parquet')}
save('dataset-validation.json', {'status': 'FRESH_SHA_PARQUET_PARSE_PASS', 'rows': rows})
retrieval = verify_retrieval(ROOT, c['retrieval_asset_binding'], c['retriever_command'][3])
save('retrieval-fresh-validation.json', retrieval)
save('assets-cpu-complete.json', {'status': 'CPU_SOURCE_AND_ASSET_VALIDATION_PASS',
                                 'gpu_execution_tested': False, 'epoch': time.time()})
