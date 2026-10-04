"""Imported-main Hydra configuration regression only; no task body/GPU/assets run."""
import contextlib,hashlib,importlib,io,json,os,sys
from pathlib import Path
repo=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(repo))
# Explicit placeholders identify composition-only scope; never accepted assets.
for key,value in {'OPSD_E1_OFFICIAL_MODEL':'/CPU_ONLY_NOT_ASSET_PROOF/official-aa8','OPSD_E1_COMMON_MANIFEST':'/CPU_ONLY_NOT_ASSET_PROOF/common.json','OPSD_E1_COMMON_DATA_ROOT':'/CPU_ONLY_NOT_ASSET_PROOF/common','SEARCH_R1_BASELINE_RUN_ROOT':'/CPU_ONLY_NOT_ASSET_PROOF/run','OPSD_E1_REVIEW_PUBLIC_KEY':'/CPU_ONLY_NOT_ASSET_PROOF/public.json','OPSD_E1_REVIEW_PUBLIC_KEY_SHA256':'a'*64,'OPSD_E1_BINDING_MANIFEST':'/CPU_ONLY_NOT_ASSET_PROOF/binding.json','OPSD_E1_BINDING_MANIFEST_SHA256':'b'*64}.items():os.environ.setdefault(key,value)
import torch
from omegaconf import OmegaConf
assert not torch.cuda.is_initialized()
module=importlib.import_module('search_r1.baselines.main')
assert Path(module.__file__).resolve()==repo/'search_r1/baselines/main.py'
assert module.main.__wrapped__.__module__=='search_r1.baselines.main'
assert not (repo/'verl/trainer/config/__init__.py').exists()
# This is the same imported module/function object call used by bootstrap;
# --cfg deliberately makes Hydra compose without invoking GPU/task function body.
sys.argv=['search_r1.baselines.main','--config-name','ppo_opsd_e1_300','--cfg','job','--resolve']
out=io.StringIO()
with contextlib.redirect_stdout(out):module.main()
config=OmegaConf.create(out.getvalue())
from search_r1.baselines.e1_profile import validate_profile
resolved=validate_profile(config)
assert not torch.cuda.is_initialized()
print(json.dumps({'status':'IMPORTED_MAIN_HYDRA_CONFIG_PATH_CPU_PASS_NOT_GPU','scope':'Imported-module call with Hydra --cfg job --resolve; no model/assets/task body or GPU','repo':str(repo),'source_file':str(Path(module.__file__).resolve()),'source_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'module_name':module.main.__wrapped__.__module__,'config_dir':str(repo/'verl/trainer/config'),'config_dir_has_init':False,'cuda_is_initialized':False,'resolved':resolved},sort_keys=True,allow_nan=False))
