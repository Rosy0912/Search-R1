"""This group only: ten committed online rounds and actual terminal release."""
import math
from pathlib import Path
try:
    from .utils import *
except ImportError:
    from utils import *


def totals(completions,run):
    require(len(completions)==10,'NOT_READY: exactly ten group outers required')
    updates=exposure=0
    for index,value in enumerate(completions):
        require(value['accepted'] is True and value['run_id']==run and value['outer_index']==index,'Wrong ten-outer completion sequence')
        P,K=value['P'],value['K']
        require(type(P) is int and type(K) is int and 0<=K<=P<=512,'Invalid P/K')
        delta=math.ceil(K/32);updates+=delta;exposure+=K
        require(value['optimizer_updates_delta']==delta and value['optimizer_updates']==updates
            and value['cumulative_pair_exposure']==exposure and value['scheduler_advanced_updates']==0
            and value['empty_outer_no_adam'] is (K==0),'Broken Adam/tail/K0/exposure chain')
    return dict(outer_iterations=10,optimizer_updates=updates,cumulative_pair_exposure=exposure,training_questions=5120,sampled_trajectory_slots=25600)


def validate(config):
    train,training_ref=original_config(config);modules,helper_proof=frozen_helpers(config)
    modules['prepare'].validate_config(train,launch=True)
    train['_config_sha256']=training_ref['sha256'];native=modules['checkpoint']
    shared,_=checked_json(train['common_cohort_manifest'])
    shared_proof=modules['cohort_v1'].verify_from_files(shared,train['train_identity'],train['validation_identity'])
    status,_=checked_json(fileproof(Path(train['paths']['output_root'])/'status.json'))
    require(status['phase']=='TRAIN_COMPLETE' and status['run_id']==train['run_id'] and status['config_sha256']==training_ref['sha256']
        and status['outer_iterations']==10,'NOT_READY: actual group COMPLETE missing')
    terminal,tp=checked_json(config['training']['terminal'])
    require(terminal.get('accepted') is True and terminal['run_id']==train['run_id'] and terminal['config_sha256']==training_ref['sha256']
        and terminal.get('actual_exit_code')==0 and terminal.get('signal_termination_not_exit0',False) is False
        and all(type(terminal['controller'][k]) is int for k in ('pid','uid','starttick')),'Actual successful group terminal unavailable')
    core,_,scorer_source=science(config)
    root=Path(train['paths']['output_root']);values=[];proofs=[];previous=None
    for index in range(10):
        outer=root/'outers'/f'outer-{index:05d}'
        manifest,mp=checked_json(fileproof(outer/'outer-manifest.json'))
        complete,cp=checked_json(fileproof(outer/'training/outer-complete.json'))
        require(manifest['run_id']==train['run_id'] and manifest['config_sha256']==training_ref['sha256'] and manifest['outer_index']==index
            and manifest['runtime_identity']==train['runtime_manifest']['sha256'] and manifest['source_model_identity']==train['source_model_manifest']['sha256']
            and manifest['previous_complete_checkpoint']==previous and manifest['P']==complete['P'],'Outer identity/previous committed state differs')
        actual_questions,qp=rows(manifest['questions'])
        require(actual_questions==shared['cohorts'][index]['questions'],'This outer did not use the shared frozen cohort')
        cp_value,cp_manifest=checked_json(complete['checkpoint'])
        require(cp_value['contract']==native.science_contract(train) and complete['contract']==cp_value['contract']
            and cp_value['common_initial_adapter']==train['common_initial_adapter_manifest']
            and cp_value['run_id']==train['run_id'] and cp_value['config_sha256']==training_ref['sha256']
            and cp_value['optimizer_updates']==complete['optimizer_updates'] and cp_value['cursor']['outer_index']==index
            and cp_value['cursor']['next_pair_cursor']==cp_value['cursor']['usable_pair_count']==complete['K']
            and cp_value['cumulative_pair_exposure']==complete['cumulative_pair_exposure'],'Not this group raw committed outer-end')
        encoding,ep=checked_json(cp_value['training_input']['encoding']);order,op=checked_json(cp_value['training_input']['order'])
        require(encoding['K']==complete['K'] and encoding['P']==complete['P'] and encoding['sha256']==object_sha(encoding['rows'])
            and order['encoded_sha256']==encoding['sha256']==cp_value['cursor']['encoded_sha256']
            and op['sha256']==cp_value['cursor']['order_sha256'] and sorted(order['pair_indices'])==list(range(complete['K'])),'Committed native pair/order cursor differs')
        sampling,sp=checked_json(fileproof(outer/'sampling-complete.json'))
        require(sampling['accepted'] is True and sampling['run_id']==train['run_id'] and len(sampling['workers'])==8
            and sampling['exit_proof']['accepted'] is True and len(sampling['exit_proof']['child_exit_evidence'])==8
            and all(x['actual_wait_returncode']==0 for x in sampling['exit_proof']['child_exit_evidence']),'Actual eight sampling child exits incomplete')
        actual_exit,actual_exit_ref=checked_json(sampling['exit_proof']['actual_terminal'])
        require(actual_exit['run_id']==train['run_id'] and actual_exit['config_sha256']==training_ref['sha256']
            and actual_exit['phase']=='sampling-running' and actual_exit['child_exit_evidence']==sampling['exit_proof']['child_exit_evidence'],
            'Sampling actual terminal evidence differs')
        ranks=set()
        require(sampling['sampling_model_identity']['checkpoint']==previous
            and sampling['sampling_model_identity']['source_model_identity']==train['source_model_manifest']['sha256']
            and sampling['sampling_model_identity']['optimizer_updates']==(values[-1]['optimizer_updates'] if values else 0),
            'Sampling is not this outer-start latest student')
        combined=[]
        for ref in sampling['workers']:
            worker,_=checked_json(ref)
            require(worker['accepted'] is True and worker['run_id']==train['run_id'] and worker['config_sha256']==training_ref['sha256']
                and worker['variant']==train['variant'] and worker['precision']==train['precision'] and worker['rows']==320,
                'Incomplete or foreign actual512x5 sampling')
            require(worker['rank'] in range(8) and worker['rank'] not in ranks,'Duplicate sampling rank');ranks.add(worker['rank'])
            sampled,_=rows(worker['rollouts'])
            from collections import Counter
            require(len(sampled)==320 and Counter(x['question_id'] for x in sampled)==Counter({x['question_id']:5 for x in actual_questions[worker['rank']::8]}),
                'Actual sampling did not cover this rank64 questions with five siblings')
            combined.extend(sampled)
        saved_rollouts,rp=rows(sampling['rollouts']);saved_pairs,pairp=rows(sampling['pairs'])
        require(combined==saved_rollouts and manifest['rollouts']==sampling['rollouts'] and manifest['pairs']==sampling['pairs'],
            'Mined input differs from the actual eight worker outputs')
        regenerated,stats=modules['miner_v1'].verify_rollouts(saved_rollouts,actual_questions,train['variant'],index,core)
        require(regenerated==saved_pairs and stats==sampling['stats'] and stats==manifest['sampling_stats']
            and len(saved_pairs)==manifest['P'],'Versioned native strict miner/pair/weight identity differs')
        exits=[]
        for path in sorted((outer/'trainer-logs').glob('attempt-*-exit.json')):
            value,fp=checked_json(fileproof(path))
            if value.get('accepted') is True:
                require(value['child_exit_evidence'] and all(x['actual_wait_returncode']==0 for x in value['child_exit_evidence']),'Committed trainer actual wait nonzero')
                terminal_value,_=checked_json(value['actual_terminal'])
                require(terminal_value['run_id']==train['run_id'] and terminal_value['config_sha256']==training_ref['sha256']
                    and terminal_value['phase']=='training-running' and terminal_value['child_exit_evidence']==value['child_exit_evidence'],
                    'Trainer actual terminal was borrowed from another phase/run')
                exits.append(fp)
        require(exits,'No actual successful trainer exit proof for outer')
        values.append(complete);previous=complete['checkpoint'];proofs.append(dict(outer=index,manifest=mp,completion=cp,checkpoint=cp_manifest,encoding=ep,order=op,sampling=sp,
            sampled_rollouts=rp,exact_mined_pairs=pairp,versioned_miner_stats=stats,sampling_actual_terminal=actual_exit_ref,trainer_exits=exits))
    total=totals(values,train['run_id'])
    require(total['optimizer_updates']>=5,'Five real updates have not occurred; not complete experimental acceptance')
    require(status['actual_optimizer_updates']==total['optimizer_updates'] and status['latest_checkpoint']==previous
        and config['training']['raw10_checkpoint']==previous,'Selected checkpoint is not this group raw tenth-outer endpoint')
    final,actual=native.check_checkpoint(previous,train,train['run_id'],total['optimizer_updates'],9)
    require(final['cumulative_pair_exposure']==total['cumulative_pair_exposure'],'Raw10 exposure differs')
    # Actual adapter finite/count/change is independently read again by every
    # admitted worker; native file hashes here are not Adam tensor semantics.
    return dict(schema='opsd-ablation-ten-outer-eval-readiness-v1',accepted=True,study_id=config['study_id'],variant=config['variant'],
        training_run_id=train['run_id'],training_config=training_ref,terminal=tp,totals=total,final_checkpoint=previous,
        final_checkpoint_fresh_files=actual,outer_proofs=proofs,helper_source=helper_proof,scorer_source=scorer_source,common_cohorts=shared_proof,
        scope='Actual10 rounds, persisted child rc0 and native full file SHA; no claim that scalar bookkeeping independently proves optimizer tensor semantics')
