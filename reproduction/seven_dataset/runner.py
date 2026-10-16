"""Run native greedy search on the full split with independent prediction capture."""
import json
import os
import time
from pathlib import Path
from .contracts import TOTAL,DATA_SHA,validate_expected_rows,accept_rows,require,sha


def worker_setup():
    from reproduction.seven_dataset.bootstrap import install
    install()


def run(config,plan,expected):
    import ray
    from verl.utils import hf_tokenizer
    from verl.single_controller.ray import RayWorkerGroup
    from verl.trainer.main_ppo import RewardManager,decode_model_response_spans
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer,ResourcePoolManager,Role
    from search_r1.baselines.bapo import answer_from_spans
    from search_r1.baselines.holdout import ordered_question_ids_hash,frame_identity,plain
    from search_r1.baselines.tree_raw_tokens import register_validation_tokenizer,tokenizer_validation_scope,semantic_spans
    from verl.utils.reward_score.qa_em import em_check
    import search_r1.baselines.evaluation as scoring
    from .worker import FrozenWorker
    out=Path(plan['output']);rows_path=out/'predictions.jsonl'
    validate_expected_rows(expected)
    tokenizer=hf_tokenizer(config.actor_rollout_ref.model.path)
    trainer=RayPPOTrainer(config=config,tokenizer=tokenizer,
        role_worker_mapping={Role.ActorRollout:ray.remote(FrozenWorker)},
        resource_pool_manager=ResourcePoolManager(resource_pool_spec={'seven':[8]},mapping={Role.ActorRollout:'seven'}),
        ray_worker_group_cls=RayWorkerGroup,reward_fn=RewardManager(tokenizer,0),val_reward_fn=RewardManager(tokenizer,0))
    trainer.init_workers();trainer._prepare_final_validation()
    require(trainer.actor_rollout_wg.world_size==8,'Eight evaluation workers required')
    require(len(trainer.val_dataset)==TOTAL,'Final dataset was truncated')
    require(ordered_question_ids_hash(trainer.val_dataset.dataframe)==plan['ordered_ids_sha256'],'Loaded question order differs')
    require(frame_identity(trainer.val_dataset.dataframe,config.data.prompt_key)==plan['final_content_identity'],
            'Loaded final frame content differs')
    trainer.global_steps=plan['outer'];trainer.actual_outer_updates=plan['outer']
    binding=register_validation_tokenizer(tokenizer)
    original_reward=trainer.val_reward_fn;original_score=scoring.batch_answer_metrics
    context=None;completed=0;started=time.time()
    rows_path.open('x').close()
    def reward(batch):
        nonlocal context
        with tokenizer_validation_scope(tokenizer,binding) as active:result=original_reward(batch)
        context=active
        return result
    def record(batch,tok):
        nonlocal context,completed
        require(tok is tokenizer and context is not None,'Scoring outside bound validation batch')
        saved=[]
        with tokenizer_validation_scope(tokenizer,binding,_context=context):
            metrics=original_score(batch,tok)
            for j,row in enumerate(metrics):
                item=batch[j];width=item.batch['responses'].shape[-1]
                mask=item.batch['attention_mask'][-width:].bool()
                if 'info_mask' in item.batch:mask=mask & item.batch['info_mask'][-width:].bool()
                spans=semantic_spans(item,tok)
                if spans is None:spans=decode_model_response_spans(tok,item.batch['responses'],mask)
                prediction=answer_from_spans(spans)
                gold=plain(item.non_tensor_batch['reward_model']['ground_truth']['target'])
                if isinstance(gold,str):gold=[gold]
                index=completed+j;want=expected[index]
                question_id=plain(item.non_tensor_batch['index'])
                require(question_id==want['question_id'] and row['source']==want['source'] and gold==want['gold_answers'],
                        'Generated batch does not match fixed question/gold identity')
                em=int(bool(em_check(prediction,gold))) if prediction is not None else 0
                require(row['em']==em,'Native EM differs from captured prediction')
                saved.append(dict(row,final_index=index,question_id=question_id,prediction=prediction,
                    gold_answers=gold,model_response_spans=spans,checkpoint_identity=plan['checkpoint_identity']))
        context=None
        with rows_path.open('a') as f:
            for row in saved:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            f.flush();os.fsync(f.fileno())
        completed+=len(saved)
        progress=dict(completed=completed,total=TOTAL,time=time.time(),elapsed_seconds=time.time()-started,
            checkpoint_identity=plan['checkpoint_identity'],job=plan['job'])
        tmp=out/'progress.tmp';tmp.write_text(json.dumps(progress));tmp.replace(out/'progress.json')
        print('SEVEN_EVAL_PROGRESS='+json.dumps(progress),flush=True)
        return metrics
    trainer.val_reward_fn=reward;scoring.batch_answer_metrics=record
    try:
        native=trainer._validate()
        require(completed==TOTAL and native['val/qa_count/all']==TOTAL,'Native full evaluation incomplete')
        rows=[json.loads(line) for line in rows_path.read_text().splitlines()]
        report=accept_rows(rows,expected,plan['checkpoint_identity'],em_check)
        require(abs(report['all_question_em']-native['val/em/all'])<1e-12,'Aggregate native EM differs')
        workers=trainer.actor_rollout_wg.seven_frozen_evidence()
        if plan.get('communication_transport'):
            require(all(w['communication_transport']==plan['communication_transport'] for w in workers),
                    'Workers did not inherit the verified communication transport')
        require(len(workers)==8 and {w['rank'] for w in workers}==set(range(8)) and
                len({w['gpu_uuid'] for w in workers})==8 and
                all(w['checkpoint_identity']==plan['checkpoint_identity'] and w['calls']>0 and
                    w['optimizer_states']==w['trainable_parameters']==0 for w in workers),
                'Frozen inference execution evidence invalid')
        trainer.actor_rollout_wg.release_rollout_policy_cache()
        return dict(report,native_metrics=native,workers=workers,elapsed_seconds=time.time()-started,
                    raw_emitted_tokens=sum(w['raw_emitted_tokens'] for w in workers),
                    predictions_sha256=sha(rows_path))
    finally:
        scoring.batch_answer_metrics=original_score;trainer.val_reward_fn=original_reward
