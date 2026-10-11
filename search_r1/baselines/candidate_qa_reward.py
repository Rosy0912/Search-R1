"""Opt-in recursive candidate QA reward without assembling policy tensors.

Final training compose/probability validation remains unchanged. Candidate
semantics are rebuilt with the existing original-token/ByteLevel contract.
"""
from .tree_raw_tokens import (_trajectory_values, tokenizer_validation_scope,
                              require, tokenizer_identity)


def _decoded_model_spans(tokens, marks, tokenizer):
    spans=[];start=None
    for index,is_model in enumerate(marks):
        if is_model and start is None:start=index
        elif not is_model and start is not None:
            spans.append(tokenizer.decode(tokens[start:index]));start=None
    if start is not None:spans.append(tokenizer.decode(tokens[start:]))
    return spans


def candidate_qa_rewards(engine, qa_batch, states, reward_fn):
    from verl.trainer.main_ppo import RewardManager, _select_rm_score_fn
    from verl.utils.reward_score import qa_em
    require(type(reward_fn) is RewardManager,
            'Light candidate scoring requires the unchanged QA RewardManager')
    require(tokenizer_identity(reward_fn.tokenizer,
                engine.tokenizer_identity['tokenizer_json_sha256'])==engine.tokenizer_identity,
            'Light candidate reward tokenizer differs from the frozen complete identity')
    require(reward_fn.format_score==0. and engine.token_preservation is not None,
            'Light candidate scoring requires strict binary QA and raw-token preservation')
    require(engine.config.max_prompt_length==4096,
            'Light candidate scoring requires the frozen semantic response cap')
    contexts={};scores=[]
    with tokenizer_validation_scope(engine.tokenizer,engine.tokenizer_identity):
        for state in states:
            source=state.source
            require(type(source) is int and 0<=source<len(qa_batch),
                    'Candidate source is outside the original question batch')
            source_name=qa_batch.non_tensor_batch['data_source'][source]
            score_fn=_select_rm_score_fn(source_name)
            require(score_fn is qa_em.compute_score_generated_em,
                    'Light candidate scoring cannot substitute a different reward function')
            value=_trajectory_values(state,engine.config.max_prompt_length,contexts)
            spans=_decoded_model_spans(value['semantic_response_token_ids'],
                                      value['semantic_model_mask'],engine.tokenizer)
            # RewardManager leaves an empty-response row at exactly zero.
            score=(score_fn(model_spans=spans,
                 ground_truth=qa_batch.non_tensor_batch['reward_model'][source]['ground_truth'],
                 format_score=reward_fn.format_score) if len(state.response) else 0.)
            require(type(score) in (int,float) and score in (0.,1.),
                    'Candidate strict QA reward is not binary')
            scores.append(float(score))
    return scores
