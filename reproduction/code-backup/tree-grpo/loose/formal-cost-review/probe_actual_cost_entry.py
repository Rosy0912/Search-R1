import hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];work=ROOT/'work';repo=work/'tree-grpo';sys.path[:0]=[str(work),str(repo)]
import tree_shared_supervisor_formal_cost as formal
from search_r1.baselines import holdout
from omegaconf import OmegaConf
with tempfile.TemporaryDirectory() as temp:
 root=Path(temp);(root/'repo').symlink_to(repo,target_is_directory=True);run=root/'tree-formal-1';run.mkdir()
 def split(v):
  v.data.train_files=str(root/'split/train.parquet');v.baselines.bapo.stage_validation_files=str(root/'split/stage-validation.parquet')
  return {'content_identity':{'sha256':formal.CONTENT},'holdout_rows':3393}
 with patch.object(formal,'ROOT',root),patch.object(holdout,'prepare_common_holdout',split):formal.configuration(run,'formal')
 env=dict(os.environ,PYTHONPATH=str(repo)+':'+str(work),SEARCH_R1_ASSETS='/read-only-fixture-assets')
 oldargs=['baselines.method=tree_grpo','trainer.n_gpus_per_node=8','actor_rollout_ref.rollout.n_agent=2','--config-dir',str(run),'--config-name','input-config','--cfg','job']
 original=subprocess.run([sys.executable,'-m','search_r1.baselines.main',*oldargs],cwd=repo,env=env,text=True,capture_output=True);assert original.returncode==0,original.stderr
 broken=subprocess.run([sys.executable,str(work/'validation_cost_entry.py'),*oldargs],cwd=repo,env=env,text=True,capture_output=True);assert broken.returncode!=0 and 'Primary config module' in broken.stderr
 newargs=[x if x!='--config-dir' else '--config-path' for x in oldargs]
 fixed=subprocess.run([sys.executable,str(work/'validation_cost_entry.py'),*newargs],cwd=repo,env=env,text=True,capture_output=True);assert fixed.returncode==0,fixed.stderr
 assert original.stdout==fixed.stdout
 # Apply identical holdout relocation to the actual inspected entry config: exact reviewed canonical hash.
 cfg=OmegaConf.create(fixed.stdout);split(cfg);canonical=OmegaConf.to_yaml(OmegaConf.create(OmegaConf.to_container(cfg,resolve=True)))
 assert canonical==(run/'resolved.yaml').read_text()
 result={'status':'ACTUAL_IMPORTED_ENTRY_CONFIG_DIR_REJECTED_ABSOLUTE_CONFIG_PATH_EQUAL','old_entry_returncode':original.returncode,'candidate_config_dir_returncode':broken.returncode,'config_dir_error':broken.stderr,'absolute_config_path_returncode':fixed.returncode,'old_and_absolute_cfg_byte_equal':True,'post_holdout_resolved_byte_equal':True,'canonical_sha256':hashlib.sha256(canonical.encode()).hexdigest(),'scope':'Actual CPU Hydra processes with production formal config, mocked holdout relocation; noGPU/assets/Slurm/formal execution'}
 print(json.dumps(result,indent=2))
