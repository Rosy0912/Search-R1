"""Actual closed training prefix; no ten-only/best/cache assumptions."""
import math
from pathlib import Path
try:
    from .utils import *
except ImportError:
    from utils import *

def validate(config):
    train,tf=original_config(config);n=config['completed_outers'];modules,hp=frozen_helpers(config)
    modules['prepare'].validate_config(train,launch=True);native=modules['checkpoint']
    terminal,tp=checked_json(config['training']['terminal']);status,sp=checked_json(terminal['status'])
    require(terminal['accepted'] is True and terminal['actual_exit_code']==0 and terminal['run_id']==train['run_id']
        and terminal['config_sha256']==tf['sha256'] and terminal['completed_outers']==n,'Training actual foreground0 missing')
    require(status['phase']==('TRAIN_COMPLETE' if n==100 else 'TRAIN_BOUNDARY') and status['run_id']==train['run_id']
        and status['config_sha256']==tf['sha256'] and status['completed_outers']==n
        and status['outer_iterations']==100 and status['owner']==terminal['controller'],'Actual training boundary differs')
    actual_exit,_=checked_json(terminal['actual_child_exit']);release,_=checked_json(terminal['actual_release'])
    require(actual_exit['actual_wait_returncode']==0 and actual_exit['identity']==terminal['controller'] and release['accepted'] is True,'Parent wait/release differs')
    boundary,bp=checked_json(config['training']['boundary'])
    require(boundary['accepted'] is True and boundary['run_id']==train['run_id'] and boundary['config_sha256']==tf['sha256']
        and boundary['completed_outers']==n and boundary['checkpoint']==config['training']['boundary_checkpoint'],'Immutable boundary checkpoint differs')
    root=Path(train['paths']['output_root']); previous=None;updates=exposure=0;proofs=[]
    for index in range(n):
        outer=root/'outers'/f'outer-{index:05d}'
        manifest,mp=checked_json(fileproof(outer/'outer-manifest.json'));value,cp=checked_json(fileproof(outer/'training/outer-complete.json'))
        require(manifest['run_id']==value['run_id']==train['run_id'] and manifest['config_sha256']==tf['sha256']
            and value['accepted'] is True and value['outer_index']==index and manifest['outer_index']==index
            and manifest['previous_complete_checkpoint']==previous,'Incomplete/wrong committed prefix')
        P,K=value['P'],value['K'];require(type(P) is int and type(K) is int and 0<=K<=P<=512,'Invalid P/K')
        updates+=math.ceil(K/32); exposure+=K
        require(value['optimizer_updates_delta']==math.ceil(K/32) and value['optimizer_updates']==updates
            and value['cumulative_pair_exposure']==exposure and value['scheduler_advanced_updates']==0
            and value['empty_outer_no_adam'] is (K==0),'Adam/tail/exposure prefix differs')
        previous=value['checkpoint'];proofs.append(dict(outer_index=index,manifest=mp,completion=cp,checkpoint=previous))
    require(previous==config['training']['boundary_checkpoint']==status['latest_checkpoint']
        and status['actual_optimizer_updates']==updates,'Wrong selected raw endpoint')
    final,fresh=native.check_checkpoint(previous,train,train['run_id'],updates,n-1)
    require(final['cumulative_pair_exposure']==exposure and final['cursor']['next_pair_cursor']==final['cursor']['usable_pair_count'],'Boundary checkpoint/cursor differs')
    return dict(schema='opsd-pr100-boundary-readiness-v1',accepted=True,training_config=tf,training_run_id=train['run_id'],completed_outers=n,
        terminal=tp,boundary=bp,status=sp,totals=dict(outer_iterations=n,optimizer_updates=updates,cumulative_pair_exposure=exposure,
            training_questions=n*512,sampled_trajectory_slots=n*2560),final_checkpoint=previous,final_checkpoint_fresh_files=fresh,
        outer_proofs=proofs,helper_source=hp,scope='Native committed prefix/actual parent rc0 and resource release; no independent Adam tensor-semantics claim')
