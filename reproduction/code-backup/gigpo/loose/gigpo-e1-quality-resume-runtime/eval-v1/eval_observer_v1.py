"""Read-only validation observer; call the frozen original main_task body."""
import gzip, json, os
from pathlib import Path
import ray


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial=path.with_name(path.name+'.partial')
    assert not path.exists() and not partial.exists()
    with partial.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as stream:
            for row in rows:stream.write((json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n').encode())
        raw.flush();os.fsync(raw.fileno())
    partial.rename(path)


@ray.remote
def observed_main_task(config):
    from search_r1.llm_agent.generation import LLMGenerationManager
    from verl.trainer.main_ppo import main_task, RewardManager
    from search_r1.baselines.evaluation import batch_answer_metrics
    root=Path(config.trainer.default_local_dir).parent/'trajectories'
    original_execute=LLMGenerationManager.execute_predictions
    original_compose=LLMGenerationManager._compose_final_output
    original_reward=RewardManager.__call__
    qa_count=[0]

    def execute(self, predictions, pad_token, active_mask=None, do_search=True):
        result=original_execute(self,predictions,pad_token,active_mask,do_search)
        observations,dones,valid,search=result
        batch=getattr(self,'_quality_trace_batch',0)
        turn=getattr(self,'_quality_trace_turn',0)+1
        self._quality_trace_turn=turn
        write_rows(root/f'batch_{batch:03d}'/f'turn_{turn:02d}.jsonl.gz',[
            dict(row=i,turn=turn,active=bool(active_mask[i]),model_action=predictions[i],
                 raw_tool_observation_before_token_limit=observations[i],done=bool(dones[i]),
                 valid_action=bool(valid[i]),is_search=bool(search[i])) for i in range(len(predictions))])
        return result

    def compose(self,*args,**kwargs):
        result=original_compose(self,*args,**kwargs)
        self._quality_trace_batch=getattr(self,'_quality_trace_batch',0)+1
        self._quality_trace_turn=0
        return result

    def reward(self,data):
        scores=original_reward(self,data)
        metrics=batch_answer_metrics(data,self.tokenizer)
        rows=[]
        for i in range(len(data)):
            row=dict(row=i,validation_row=qa_count[0]+i,
                     question_id=str(data.non_tensor_batch['index'][i]),metrics=metrics[i])
            for key in ('prompts','responses','attention_mask','info_mask'):
                if key in data.batch:row[key]=data.batch[key][i].detach().cpu().tolist()
            row['prompt_text']=self.tokenizer.decode(row['prompts'],skip_special_tokens=False)
            row['response_with_observations_text']=self.tokenizer.decode(row['responses'],skip_special_tokens=False)
            row['score']=float(scores[i].sum().detach().cpu())
            rows.append(row)
        write_rows(root/f'scored_{qa_count[0]:04d}.jsonl.gz',rows)
        qa_count[0]+=len(rows)
        return scores

    assert config.trainer.val_only and not config.trainer.instruct300.enabled
    assert main_task._function.__module__=='verl.trainer.main_ppo' and main_task._function.__name__=='main_task'
    LLMGenerationManager.execute_predictions=execute
    LLMGenerationManager._compose_final_output=compose
    RewardManager.__call__=reward
    try:
        result=main_task._function(config)
        assert qa_count[0]==512
        with (root/'complete.json').open('x') as f:json.dump(dict(actual_scored_rows=qa_count[0],training_updates=0),f)
        return result
    finally:
        LLMGenerationManager.execute_predictions=original_execute
        LLMGenerationManager._compose_final_output=original_compose
        RewardManager.__call__=original_reward
