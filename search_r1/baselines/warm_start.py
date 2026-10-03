"""Explicit supervised initialization; never an RL update or acceptance milestone."""
import torch
import torch.nn.functional as F
from .cosearch import contains_answer, answers_are_trivial


def training_ranking(documents, aliases, keep=3):
    if len(documents) != 50:
        raise ValueError('Warm-start requires all 50 original candidates')
    if not 1 <= keep <= 50:
        raise ValueError("Invalid selection size")
    no_answer = any(str(a).strip().lower() == 'noanswer' for a in aliases)
    relevant = [] if answers_are_trivial(aliases) or no_answer else [i + 1 for i, doc in enumerate(documents)
                if contains_answer(doc['document']['contents'], aliases)]
    ids = (relevant + [i for i in range(1, 51) if i not in relevant])[:keep]
    return ids


class SupervisedSequenceLoss(torch.nn.Module):
    """Response-only teacher-forced NLL, equal sequence weights at temperature1."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids, attention_mask, response_lengths, max_response_length):
        positions = attention_mask.cumsum(-1) - 1
        positions.masked_fill_(attention_mask == 0, 1)
        logits = self.model(ids, attention_mask=attention_mask, position_ids=positions,
                            use_cache=False, num_logits_to_keep=max_response_length + 1).logits[:, :-1]
        logp = F.log_softmax(logits.float(), -1).gather(
            -1, ids[:, -max_response_length:, None]).squeeze(-1)
        mask = torch.arange(max_response_length, device=ids.device)[None] >= (
            max_response_length - response_lengths[:, None])
        if not torch.isfinite(logp).all() or (response_lengths <= 0).any():
            raise ValueError('Invalid supervised response scores')
        return -(logp * mask).sum(-1) / response_lengths
