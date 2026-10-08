"""Passive selected-logit/LSE observations only; not real HF/vLLM GPU proof."""
import torch
from decisive_normalizer_v1 import selected_statistics


def test_selected_target_logit_and_fp32_lse_use_actual_labels_and_bounded_vectors():
    logits=torch.arange(36,dtype=torch.float32).reshape(4,9)/10
    labels=torch.tensor([1,2,3,4]); state=torch.get_rng_state().clone()
    budget=[2]; result=selected_statistics(logits,labels,[3,1],vector_budget=budget)
    assert result['target_ids'].tolist()==[4,2]
    assert torch.equal(result['pre_normalization_selected_logits_fp32'],logits[[3,1],[4,2]])
    assert torch.equal(result['fp32_logsumexp'],torch.logsumexp(logits[[3,1]],-1))
    assert len(result['bounded_full_vocab_vectors'])==2 and budget==[0]
    assert torch.equal(state,torch.get_rng_state())


def test_empty_mask_adds_no_vector_or_probability_mass():
    result=selected_statistics(torch.zeros(2,9),torch.zeros(2,dtype=torch.long),[],vector_budget=[6])
    assert result['target_ids'].numel()==result['fp32_logsumexp'].numel()==0
    assert result['bounded_full_vocab_vectors']==[]
