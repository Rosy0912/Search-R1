"""Exercise actual imported-main Hydra config path, never task body/CUDA."""
import os,subprocess,sys,json
from pathlib import Path
import pytest
from omegaconf import OmegaConf
ROOT=Path(__file__).resolve().parents[2]
def env():return dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1')
def test_actual_imported_main_cfg_resolve_without_package_init():
 p=subprocess.run([sys.executable,str(ROOT/'reproduction/baselines/e1-300/cpu_hydra_entry_regression.py')],env=env(),capture_output=True,text=True)
 assert p.returncode==0,p.stderr
 r=json.loads(p.stdout);assert r['status']=='IMPORTED_MAIN_HYDRA_CONFIG_PATH_CPU_PASS_NOT_GPU' and r['cuda_is_initialized'] is False and r['config_dir_has_init'] is False
 d=r['resolved'];assert d['actor_rollout_ref']['rollout']['n_agent']==5 and d['actor_rollout_ref']['actor']['optim']['lr_warmup_steps']==57 and d['trainer']['total_training_steps']==300
 assert d['baselines']['distillation']['component_clip'] is None

def test_exact_old_imported_entry_reproduces_real_pkg_failure():
 # Compile exact preserved Git bytes into imported module identity. No file edit.
 old=subprocess.check_output(['git','-C',str(ROOT),'show','60cbea97d61ff95a8f7dfa58de7b79962e79bd10:search_r1/baselines/main.py'])
 code='''import sys,importlib,subprocess
from pathlib import Path
import torch
repo=Path(sys.argv[1]);sys.path.insert(0,str(repo));m=importlib.import_module('search_r1.baselines.main')
raw=subprocess.check_output(['git','-C',str(repo),'show','60cbea97d61ff95a8f7dfa58de7b79962e79bd10:search_r1/baselines/main.py'])
exec(compile(raw,str(repo/'search_r1/baselines/main.py'),'exec'),m.__dict__)
assert not torch.cuda.is_initialized()
sys.argv=['search_r1.baselines.main','--config-name','ppo_opsd_e1_300','--cfg','job'];m.main()
'''
 p=subprocess.run([sys.executable,'-c',code,str(ROOT)],env=env(),capture_output=True,text=True)
 assert p.returncode!=0
 assert "Primary config module 'verl.trainer.config' not found" in p.stderr
