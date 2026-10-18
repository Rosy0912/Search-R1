import hashlib
import re
import shlex
from pathlib import Path

UPSTREAM = '19bf3fa0c74d8b9ba70619a09a2fd480c31b9d59'
MODEL_REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
MANIFEST_SHA = '7b6369e17ebd497ea159d57dc159647eed8adad852b7468cc600b7f203e80f43'
JOB = 'zhangjunming-ttt-treegrpo-fair-v1'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()

def recipe(script, model, data, output, retriever):
    """Take every upstream CLI override; change only agreed inputs and local execution."""
    values = {'n_gpus_per_node':'8','train_batch_size':'512','val_batch_size':'128',
              'actor_ppo_mini_batch_size':'64','actor_ppo_micro_batch_size':'32',
              'log_prob_micro_batch_size':'32','tree_search_m':'2','tree_search_n':'2',
              'tree_search_l':'1','tree_search_k':'3','lr':'1e-6','BASE_MODEL':str(model),
              'DATA_DIR':str(data),'WAND_PROJECT':'zhangjunming-ttt',
              'EXPERIMENT_NAME':JOB,'BASE_DIR':str(output)}
    result = {}
    for line in script.splitlines():
        line = line.strip().removesuffix('\\').strip()
        if not re.match(r'\+?[a-z][a-z_]*(\.[a-z_]+)+=', line): continue
        key, value = shlex.split(line)[0].split('=',1)
        value = re.sub(r'\$\{(\w+)\}|\$(\w+)', lambda m:values[m[1] or m[2]], value)
        assert '$' not in value
        result[key] = value
    # max_turns has no dot and is therefore collected explicitly.
    assert re.search(r'^\s*max_turns=3\s*\\', script, re.M)
    result['max_turns']='3'
    result['data.val_files']=str(Path(data)/'validation.parquet')
    result['trainer.default_local_dir']=str(Path(output)/'checkpoints')
    result['trainer.logger']="['console']"
    result['retriever.url']=retriever.rstrip('/')+'/retrieve'
    assert result['algorithm.adv_estimator']=='tree'
    assert result['actor_rollout_ref.rollout.tree_search']=='true'
    assert result['trainer.total_training_steps']=='180'
    assert result['actor_rollout_ref.actor.optim.lr']=='1e-6'
    return [k+'='+v for k,v in result.items()]

def patch_observation(root, observer):
    """Add post-event observation only. Neither updates nor rollout math are replaced."""
    root=Path(root)
    specs=[('verl/workers/actor/dp_actor.py',
        '            self.actor_optimizer.step()\n',
        '            self.actor_optimizer.step()\n            from treegrpo_observe import observe_adam\n            observe_adam(self.actor_optimizer, grad_norm)\n'),
        ('verl/trainer/ppo/ray_trainer_ts.py',
        '                logger.log(data=metrics, step=self.global_steps)\n',
        '                logger.log(data=metrics, step=self.global_steps)\n                from treegrpo_observe import observe_iteration\n                observe_iteration(self.global_steps, metrics)\n')]
    records=[]
    for name,old,new in specs:
        p=root/name;s=p.read_text();assert s.count(old)==1, name
        before=sha(p);p.write_text(s.replace(old,new));records.append(dict(file=name,before=before,after=sha(p)))
    p=root/'treegrpo_observe.py';assert not p.exists();p.write_bytes(Path(observer).read_bytes())
    return records
