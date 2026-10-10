"""Adapt only an idle isolated runtime to the explicit four-rank protocol."""
import re,ast,hashlib,json,pathlib,subprocess
from configure_topd200 import configure
BASE=pathlib.Path('/data/user/jxu957/experiments/topd-20261008')
ROOT=BASE/'runtime';RT=ROOT/'tools/topd-instruct300-launch-v6/runtime'
def main():
    jobs=subprocess.check_output(['squeue','-u','mou176','-h','-o','%j'],universal_newlines=True)
    if any('topd' in j.lower() for j in jobs.splitlines()):raise RuntimeError('Active TOP-D job: refuse runtime mutation')
    paths=list(RT.glob('*.py'))+list((ROOT/'tools/topd-vllm-teacher-v3/topd_vllm_teacher').glob('*.py'))
    paths += [ROOT/'tools/topd-instruct300-source-v1/search_r1/baselines/controller.py',ROOT/'tools/topd-instruct300-source-v1/verl/trainer/ppo/ray_trainer.py']
    changes=[]
    for p in paths:
        backup=p.with_suffix(p.suffix+'.before-topd200')
        old=backup.read_text() if backup.exists() else p.read_text();s=old
        # Only rank-count expressions and protocol labels; sizes such as 8192 stay intact.
        for a,b in [('range(8)','range(4)'),('validate_inherited_gpu_leases(8)','validate_inherited_gpu_leases(4)'),('tensor_parallel_size=8','tensor_parallel_size=4'),('tp8','tp4'),('TP8','TP4'),('ALL8','ALL4'),('topd300','topd200'),("expected_outer_steps']!=300","expected_outer_steps']!=200"),("expected_outer_steps')!=300","expected_outer_steps')!=200"),('return 300','return 200'),('step>300','step>200'),("'trainer.total_training_steps':'301'","'trainer.total_training_steps':'201'"),("'actor_rollout_ref.actor.use_kl_loss':'true'","'actor_rollout_ref.actor.use_kl_loss':'false'"),("'actor_rollout_ref.actor.kl_loss_coef':'0.001'","'actor_rollout_ref.actor.kl_loss_coef':'0'"),('local_rows\']!=320','local_rows\']!=640')]:s=s.replace(a,b)
        s=re.sub(r"([!=]=)8(?![0-9])",r"\g<1>4",s)
        ast.parse(s)
        if s!=old:
            backup=p.with_suffix(p.suffix+'.before-topd200')
            if not backup.exists():backup.write_text(old)
            p.write_text(s);changes.append({'path':str(p),'before':hashlib.sha256(old.encode()).hexdigest(),'after':hashlib.sha256(s.encode()).hexdigest()})
    draft=configure(json.loads((BASE/'outputs/training-preparation/hpc3-config-draft.json').read_text()))
    out=BASE/'outputs/training-preparation'
    (out/'hpc3-topd200-protocol-draft.json').write_text(json.dumps(draft,indent=2))
    (out/'topd200-runtime-adaptation.json').write_text(json.dumps({'status':'RANK_PROTOCOL_ADAPTED_MANIFESTS_AND_REMAINING_GATES_PENDING','changes':changes,'gpu_job_submitted':False},indent=2))
    print('Adapted source files:',len(changes))
if __name__=='__main__':main()
