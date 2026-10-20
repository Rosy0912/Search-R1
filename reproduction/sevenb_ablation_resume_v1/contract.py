"""Execution namespaces only; original e265 science and full native loaders stay unchanged."""
import copy,json,re
from pathlib import Path
from reproduction.sevenb.ablations import GROUPS,ALLOCATION,PYTHON,COMPILER,sha
SCIENCE='e265dae594e8f0368f889815d7f6c4729257ce6c'
BASE=Path('/plm-shared/zhangjunming/Workspace/Recursive-Tree-GRPO-7B-Ablations-20261020-recovery-v1')
STEPS={'C1-root-only':33,'C2-random-final':33,'budget-5':39,'budget-20':26,'generation-2':33}
def job(group):
 if group not in GROUPS:raise ValueError('Only the existing five conditions')
 return 'zhangjunming-7b-abl-'+GROUPS[group]+'-rec1'
def validate(plan,parent):
 g=plan['group'];assert g in GROUPS and plan['job']==job(g) and plan['root']==str(BASE/'runs'/g)
 assert plan['source_commit']==SCIENCE and plan['allocation']==ALLOCATION and plan['actual_update_limit']==150 and plan['scheduler_scale']==200
 assert plan['protocol']=='sevenb_ablation_full_native_storage_recovery_v1' and not plan['fresh_from_common_initialization'] and plan['no_auto_retry'] and plan['no_hostname_affinity']
 assert plan['condition']==parent['condition'] and plan['model']==parent['model'] and plan['model_files']==parent['model_files'] and plan['data_identity']==parent['data_identity']
 assert plan['parent']['step']==STEPS[g] and plan['parent']['manifest']['identity']['completed_step']==STEPS[g]
 m=plan['parent']['manifest'];assert m['identity']['code_commit']==SCIENCE and m['identity']['job_id']==parent['job'] and m['identity']['actual_update_limit']==150 and m['driver']['next_step']==STEPS[g]+1
 assert len(m['worker_states'])==8 and {r['identity']['rank'] for r in m['worker_states']}==set(range(8))
 assert plan['parent']['uid']==plan['superseded_uid'] and plan['parent']['state']=='FAILED'
 assert re.fullmatch('[a-f0-9]{40}',plan['entry_source_commit']) and plan['entry_source_commit']!=SCIENCE
 assert plan['source']==str(BASE/'sources'/plan['entry_source_commit']) and plan['run_id']==parent['run_id']
 assert plan['output']==str(BASE/'runs'/g/'output'/('attempt-'+plan['entry_run_id'])) and re.fullmatch('[a-f0-9]{32}',plan['entry_run_id'])
 return Path(plan['root'])
def project_config(parent,root,host):
 cfg=copy.deepcopy(parent);cfg['trainer']['default_local_dir']=str(Path(root)/'checkpoints');cfg['baselines']['local_runtime_host']=host
 return cfg
def verify(plan):
 parent=json.loads(Path(plan['parent_plan']).read_text());assert sha(plan['parent_plan'])==plan['parent_plan_sha256'];root=validate(plan,parent)
 from reproduction.sevenb.ablations import verify as original_verify
 original_verify(parent)
 for n,h in plan['entry_source_files'].items():assert sha(Path(plan['source'])/n)==h,n
 for n,h in parent['wrapper_files'].items():assert plan['entry_source_files'][n]==h,n
 p=plan['parent'];assert sha(p['manifest_path'])==p['manifest_sha256'] and json.loads(Path(p['manifest_path']).read_text())==p['manifest']
 proof=json.loads(Path(plan['cpu_acceptance']).read_text());assert sha(plan['cpu_acceptance'])==plan['cpu_acceptance_sha256']
 assert proof['status']=='LATEST_ROLLING_NATIVE_INDEPENDENT_CPU_ACCEPTED' and proof['group']==plan['group']
 accepted=proof['accepted'];assert accepted['manifest_sha256']==p['manifest_sha256'] and len(accepted['ranks'])==8 and accepted['driver']['next_step']==STEPS[plan['group']]+1
 assert sha(plan['parent_resolved'])==p['manifest']['identity']['config_sha256']
 storage=json.loads(Path(plan['storage_admission']).read_text());assert sha(plan['storage_admission'])==plan['storage_admission_sha256']
 assert storage['stable_capacity_and_fsync_atomic_rename_verified'] and storage['minimum_available_bytes']>=500000000000
 return root

def validate_gate(plan,sub,gate):
 from reproduction.sevenb.ablations import validate_gate as original
 a=original(plan,sub,gate);assert sub['superseded_uid']==plan['superseded_uid'] and sub['entry_source_commit']==plan['entry_source_commit']
 return a
