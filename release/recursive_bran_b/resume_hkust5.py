"""Run the authorized speed continuation from one explicit complete B boundary."""
import argparse,hashlib,json,os,socket,time
from pathlib import Path
from omegaconf import OmegaConf

def sha(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--record-sha256',required=True);a=p.parse_args()
 if os.getuid()!=1001 or socket.gethostname().split('.')[0]!='xie-lab-4':raise ValueError('Wrong B node/UID')
 if sha(a.record)!=a.record_sha256:raise ValueError('Immutable resume record changed')
 r=json.loads(a.record.read_text());repo=Path(__file__).parents[2].resolve();root=Path('/weka_3/shaoshitong/recursive-bran-b-4gpu-hkust5-20261011')
 if str(repo)!=r['source'] or (repo/'SOURCE_COMMIT').read_text().strip()!=r['source_commit']:raise ValueError('Explicit source mismatch')
 if sha(r['config'])!=r['config_sha256'] or sha(r['binding'])!=r['binding_sha256']:raise ValueError('Config/binding changed')
 if sha(r['parent_terminal'])!=r['parent_terminal_sha256']:raise ValueError('Parent terminal changed')
 terminal=json.loads(Path(r['parent_terminal']).read_text())
 if r.get('recovery_kind')=='complete8_native_rng':
  from search_r1.baselines.recursive_b_completed_resume import validate_terminal
  validate_terminal(terminal)
 else:
  if terminal['protocol']!='recursive-b-speed-parent-terminal-v1' or terminal['status'] not in ('TECHNICAL_SWITCH_ALL_OWNED_ABSENT','TECHNICAL_SWITCH_ALL_OWNED_KERNEL_EXITED'):raise ValueError('No actual technical terminal')
  if terminal['original_controller']!={'pid':401228,'uid':1001,'tick':651299742} or not terminal['all_original_owned_kernel_exited']:raise ValueError('Wrong parent owner/release')
  for identity in terminal['owned_processes']:
   if Path('/proc/%d'%identity['pid']).exists():
    raw=Path('/proc/%d/stat'%identity['pid']).read_text();tick=int(raw.rsplit(')',1)[1].split()[19])
    if tick==identity['tick']:
     state=raw.rsplit(')',1)[1].split()[0]
     if state!='Z' or list(Path('/proc/%d/fd'%identity['pid']).glob('*')):raise ValueError('Original owner still alive or has open descriptors')
 if 'retry_terminal' in r:
  from search_r1.baselines.recursive_b_retry import validate_retry_terminal
  validate_retry_terminal(r['retry_terminal'],r['retry_terminal_sha256'])
 if 'driver_fault_terminal' in r:
  from release.recursive_bran_b.capture_driver_stage_terminal import validate
  validate(r['driver_fault_terminal'],r['driver_fault_terminal_sha256'])
 out=Path(r['output']);out.relative_to(root)
 if 'child_checkpoint_dir' in json.loads(Path(r['binding']).read_text()):
  b=json.loads(Path(r['binding']).read_text())
  if b['child_run_id']!=r['run_id'] or Path(b['child_checkpoint_dir'])!=out/'checkpoints':raise ValueError('Child artifact root does not belong to this run')
 if out.exists():raise ValueError('No blind restart/overwrite of speed output')
 os.environ.update(RECURSIVE_B_CONTINUATION_BINDING=r['binding'],RECURSIVE_B_CONTINUATION_SHA256=r['binding_sha256'],
  RECURSIVE_B_OMIT_FULL_BETA='1',RECURSIVE_B_CACHE_DETOKENIZER_LENGTH='1',
  RECURSIVE_B_READY_QUESTION_FRONTIER='1',
  RECURSIVE_B_SHARED_DRIVER_MU_PROOF_CACHE='1',
  TREE_LIGHTWEIGHT_CANDIDATE_REWARD='1',SEARCH_R1_BASELINE_RUN_ID=r['run_id'],E1_RUN_ID=r['run_id'],
  SEARCH_R1_EXECUTION_BACKEND='hkust_direct',SEARCH_R1_BASELINE_COMMIT=r['source_commit'],
  SEARCH_R1_BASELINE_CONFIG_SHA256=r['config_sha256'],SEARCH_R1_BASELINE_DATA_IDENTITY=r['data_identity'],
  SEARCH_R1_MODEL_IDENTITY='Qwen/Qwen2.5-3B-Instruct@aa8e72537993ba99e69dfaafa59ed015b17504d1',
  SEARCH_R1_BASELINE_OUTPUT=str(out),SEARCH_R1_BASELINE_AUDIT_DIR=str(out/'audit'),
  SEARCH_R1_TREE_RANK_AUDIT_DIR=str(out/'rank-audit'),SEARCH_R1_TRAJECTORY_DIR=str(out/'trajectories'),
  TREE_COMMON_DATA=str(root/'common-data'),TREE_INSTRUCT_MODEL=r['model'],
  SLURM_JOB_ID='hkust-direct-'+r['run_id'],SLURM_JOB_NODELIST='xie-lab-4',RAY_ADDRESS='local')
 from search_r1.baselines.recursive_b_schedule_resume import validate_entry,parent_manifest
 cfg=OmegaConf.load(r['config']);validate_entry(cfg);parent_manifest()
 out.mkdir()
 (out/'start.json').write_text(json.dumps(dict(time=time.time(),pid=os.getpid(),uid=os.getuid(),run_id=r['run_id'],source_commit=r['source_commit'],parent_binding_sha256=r['binding_sha256'])))
 from search_r1.baselines.main import main as train
 train.__wrapped__(cfg)

if __name__=='__main__':main()
