"""Adjust teacher initialization budget after observed zero KV blocks on TP4."""
import json,pathlib,subprocess,time
from finalize_topd200_draft import BASE
if 'topd' in subprocess.check_output(['squeue','-u','mou176','-h','-o','%j'],universal_newlines=True).lower():raise RuntimeError('Active job')
p=BASE/'outputs/training-preparation/hpc3-topd200-protocol-draft.json';c=json.loads(p.read_text());before=c['teacher_gpu_memory_utilization'];c['teacher_gpu_memory_utilization']=.45
c['teacher_memory_adjustment']={'before':before,'after':.45,'reason':'TP4 actual weights15.4136GiB per rank plus retriever9.6-10.3GiB caused zero GPU KV blocks under0.25 budget','failed_job':706357,'model_precision':'bfloat16','weights_and_objective_unchanged':True,'kv_released_before_student_update_required':True,'gpu_validation_pending':True,'epoch':time.time()}
p.write_text(json.dumps(c,indent=2));print('TEACHER_MEMORY_BUDGET_UPDATED_GPU_VALIDATION_PENDING')
