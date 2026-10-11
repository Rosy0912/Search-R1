"""Create independent fresh common-EM experiment; no weights or old outputs copied."""
import ast,base64,hashlib,json,shutil,sys
from pathlib import Path
OLD=Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/cosearch-instruct200-20261010')
NEW=OLD.parent/'cosearch-instruct200-common-em-20261011'
def deploy(payload):
    if NEW.exists():raise RuntimeError('Existing new experiment; inspect instead of replacing')
    NEW.mkdir();(NEW/'tools').mkdir()
    for n,b in payload['tools'].items():
        if Path(n).name!=n:raise ValueError(n)
        (NEW/'tools'/n).write_bytes(base64.b64decode(b))
    shutil.copytree(str(OLD/'repo'),str(NEW/'repo'),symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
    p=NEW/'repo/verl/verl/trainer/ppo/search_r1_reranker_ray_trainer.py';s=p.read_text();before=hashlib.sha256(s.encode()).hexdigest()
    point='and (is_last_step or (self.global_steps in (5, 50, 100, 150, 200) or (self.global_steps == 10 and self.config.trainer.get("recovery_acceptance_at10", False))))'
    if s.count(point)!=1:raise RuntimeError('Unexpected validation gate')
    s=s.replace(point,'and self.global_steps in (5, 50, 100, 150, 200)')
    point='                logger.log(data=metrics, step=self.global_steps)'
    if s.count(point)!=1:raise RuntimeError('Unexpected completed outer logging')
    s=s.replace(point,'                from common_em_runtime import completed_outer\n                metrics.update(completed_outer(self.global_steps))\n'+point)
    ast.parse(s);p.write_text(s)
    (NEW/'data').symlink_to(OLD/'data',target_is_directory=True)
    config=json.loads((OLD/'author-cli.json').read_text());config=[x.replace(str(OLD),str(NEW)).replace('trainer.experiment_name=cosearch-instruct200-20261010','trainer.experiment_name=cosearch-instruct200-common-em-20261011') for x in config]
    (NEW/'author-cli.json').write_text(json.dumps(config,indent=2)+'\n')
    storage=json.loads((OLD/'storage-authorization.json').read_text());storage['destination']='/backup/shaoshitong/cosearch-instruct200-common-em-20261011/checkpoints';storage['authority']='2026-10-11 human fresh official initialization/common EM/acd_ue'
    (NEW/'storage-authorization.json').write_text(json.dumps(storage,indent=2)+'\n')
    receipt={'old_root':str(OLD),'new_root':str(NEW),'fresh_official_initialization':True,'resume_plan':False,'validation_steps':[5,50,100,150,200],'algorithm_reward_unchanged':True,'before_trainer_sha256':before,'after_trainer_sha256':hashlib.sha256(s.encode()).hexdigest()}
    (NEW/'common-em-deployment.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt
if __name__=='__main__':print(json.dumps(deploy(json.loads(sys.stdin.read()))))
