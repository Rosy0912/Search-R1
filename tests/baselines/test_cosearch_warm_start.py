import copy
import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM
from search_r1.baselines.warm_start import SupervisedSequenceLoss, training_ranking


@pytest.mark.parametrize('rows', [1, 3, 8])
def test_response_only_supervised_loss_and_all_gradients_match_unpadded_full_logits(rows):
    torch.manual_seed(31)
    torch.set_num_threads(1)
    a = Qwen2ForCausalLM(Qwen2Config(vocab_size=37, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        attention_dropout=0., attn_implementation='sdpa')).eval()
    b = copy.deepcopy(a)
    examples = [torch.tensor([2 + (i+j)%32 for j in range(4+i)] + [0]) for i in range(rows)]
    lengths = torch.tensor([1+i%3 for i in range(rows)])
    width = max(map(len, examples))
    ids = torch.stack([F.pad(x, (width-len(x), 0), value=0) for x in examples])
    mask = torch.stack([F.pad(torch.ones_like(x), (width-len(x), 0)) for x in examples])
    actual = SupervisedSequenceLoss(a)(ids, mask, lengths, int(lengths.max())).mean()
    losses = []
    for x, length in zip(examples, lengths):
        logits = b(x[None], use_cache=False).logits[0, -length-1:-1]
        losses.append(F.cross_entropy(logits, x[-length:]))
    expected = torch.stack(losses).mean()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    actual.backward(); expected.backward()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=3e-7, rtol=3e-5)


def test_supervised_ranking_uses_training_aliases_and_preserves_all_candidates():
    docs = [dict(document=dict(contents='unrelated')) for _ in range(50)]
    docs[11] = dict(document=dict(contents='Paris is a city'))
    assert training_ranking(docs, ['Paris']) == [12, 1, 2]
    assert len(docs) == 50
    with pytest.raises(ValueError):
        training_ranking(docs[:3], ['Paris'])
