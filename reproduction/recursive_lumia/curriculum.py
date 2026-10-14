"""Native root rollouts screen questions; unchanged recursive search trains them."""
import copy
import json
import os
from pathlib import Path
import time
from .selection import Selector, PROTOCOL

def enabled(config):
    return bool(config.baselines.get('question_curriculum',{}).get('enabled',False))

def validate(config):
    from omegaconf import OmegaConf
    p=json.loads((Path(__file__).parent/'plan.json').read_text())['curriculum']
    actual=OmegaConf.to_container(config.baselines.question_curriculum,resolve=True)
    assert actual==dict(enabled=True,**p),'Curriculum protocol differs from frozen plan'
    assert config.trainer.actual_update_limit==150 and config.trainer.total_training_steps==151
    assert list(config.trainer.evaluation_steps)==[5,50,100,150]
    assert list(config.trainer.checkpoint_save_steps)==[5,50,100,150]

class CurriculumSampler:
    def __init__(self,dataset,batch_size,seed,identity):
        assert batch_size==512
        self.dataset=dataset
        self.rows,self.batch_size,self.identity,self.shuffle=len(dataset),batch_size,identity,True
        self.order,self.cursor,self.epoch=None,0,-1
        self.selector=Selector(dataset.dataframe['data_source'].tolist(),seed)
        self.trainer=self.manager=None

    def __len__(self):return self.rows//self.batch_size

    def attach(self,trainer,manager):
        self.trainer,self.manager=trainer,manager

    def screen(self,step):
        from verl import DataProto
        from verl.utils.dataset.rl_dataset import collate_fn
        from search_r1.baselines.rollout import SearchRollout
        trainer,manager=self.trainer,self.manager
        candidates=self.selector.candidates(step)
        output=Path(os.environ['SEARCH_R1_BASELINE_OUTPUT'])/'curriculum'
        output.mkdir(exist_ok=True)
        intent=output/f'screen-{step}.intent.json'
        if intent.exists():raise RuntimeError('Unknown or interrupted screen: refuse blind repetition')
        intent.write_text(json.dumps(dict(step=step,train_rows=candidates,paths=5*len(candidates),
            optimizer_updates=0,selection_before_rewards=True)))
        rewards={};cost={};start=time.perf_counter()
        for offset in range(0,len(candidates),64):
            indices=candidates[offset:offset+64]
            qa=DataProto.from_single_dict(collate_fn([self.dataset[i] for i in indices])).repeat(5,interleave=True)
            qa.meta_info.update(baseline_outer_step=step,baseline_attempt_id=getattr(trainer,'training_attempts',0)+1,
                baseline_completed_outer=trainer.actual_outer_updates,curriculum_screen=True)
            gen=qa.pop(batch_keys=['input_ids','attention_mask','position_ids'])
            tree=trainer.config.baselines.tree
            engine=SearchRollout(manager,trainer.config.baselines.gigpo.anchor_mode,None,42,'error',
                token_preservation=tree.token_preservation,behavior_mu=tree.behavior_mu)
            engine.question_uids=qa.non_tensor_batch['index'].tolist()
            engine.rollout_outer=step
            # tree=None means five independent complete roots, without acceptance or filtering.
            generated,sources=engine.run(gen,tree=None)
            assert sources.tolist()==list(range(len(indices)*5))
            scores=trainer.baseline_controller.tree_training_rewards(engine,qa,engine.last_states)
            assert len(scores)==len(indices)*5 and all(x in (0,1) for x in scores)
            for j,i in enumerate(indices):rewards[i]=[int(x) for x in scores[j*5:j*5+5]]
            for key,value in engine.cost.items():cost[key]=cost.get(key,0)+value
            with (output/f'screen-{step}.raw.jsonl').open('a') as f:
                f.write(json.dumps(dict(train_rows=indices,rewards=[rewards[i] for i in indices],cost=engine.cost))+'\n')
            del generated,gen,qa,engine
        scores=self.selector.update(step,candidates,rewards)
        record=dict(step=step,questions=len(candidates),real_root_paths=5*len(candidates),
            optimizer_updates=0,wall_seconds=time.perf_counter()-start,cost=cost,
            pool=self.selector.pool,scores=scores,uniform_fallback=self.selector.pool is None)
        (output/f'screen-{step}.complete.json').write_text(json.dumps(record))
        print('CURRICULUM_SCREEN='+json.dumps({k:v for k,v in record.items() if k not in ('pool','scores')}),flush=True)

    def __iter__(self):
        assert self.trainer is not None
        if self.order is None:self.order=[];self.epoch=0
        while self.selector.steps<150:
            step=self.trainer.global_steps
            assert step==self.selector.steps+1,'Curriculum batches must match consumed scientific rounds'
            if (step-1)%25==0:self.screen(step)
            rows=self.selector.choose()
            self.order.extend(rows);self.cursor+=len(rows)
            output=Path(os.environ['SEARCH_R1_BASELINE_OUTPUT'])/'curriculum'
            with (output/'selected-batches.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=step,train_rows=rows,refresh_step=self.selector.refresh_step,
                    uniform_fallback=self.selector.pool is None))+'\n')
            yield rows

    def state_dict(self):
        return copy.deepcopy(dict(protocol=PROTOCOL,rows=self.rows,batch_size=self.batch_size,
            identity=self.identity,shuffle=self.shuffle,order=self.order,cursor=self.cursor,epoch=self.epoch,
            selector=self.selector.state_dict()))

    def load_state_dict(self,state):
        assert state['protocol']==PROTOCOL
        for key in ('rows','batch_size','identity','shuffle'):assert state[key]==getattr(self,key)
        assert state['cursor']==state['selector']['steps']*512
        assert len(state['order'])==state['cursor']
        for offset in range(0,state['cursor'],512):
            block=state['order'][offset:offset+512]
            assert len(set(block))==512 and all(0<=i<self.rows for i in block)
        self.selector.load_state_dict(state['selector'])
        self.order,self.cursor,self.epoch=copy.deepcopy(state['order']),state['cursor'],state['epoch']
