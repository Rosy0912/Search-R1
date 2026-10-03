"""Equivalent FP32 ranker kernels: real forward, PPO gradient and greedy decode."""
import torch
from search_r1.baselines.cosearch import GenerativeRanker, sequence_ppo_loss
from tests.baselines.test_models_and_driver import TinyModels, config


def test_sdpa_matches_eager_ranker_policy_and_gradient():
    TinyModels.setUpClass()
    try:
        eager_config = config('cosearch').baselines.ranker
        eager_config.model_path = TinyModels.tmp.name
        sdpa_config = config('cosearch').baselines.ranker
        sdpa_config.model_path = TinyModels.tmp.name
        sdpa_config.attn_implementation = 'sdpa'
        eager = GenerativeRanker(eager_config)
        sdpa = GenerativeRanker(sdpa_config)
        ids = eager.tokenizer('Question Paris Rome <ranking>[1,2,3]</ranking>', return_tensors='pt')['input_ids']
        mask = torch.ones_like(ids)
        outputs, gradients, greedy = [], [], []
        for ranker in (eager, sdpa):
            ranker.model.eval()
            logits = ranker.model(ids, attention_mask=mask, use_cache=False).logits[:, -5:-1]
            logp = logits.log_softmax(-1).gather(-1, ids[:, -4:, None]).squeeze(-1)
            old = logp.detach() + .03
            loss = sequence_ppo_loss(logp[0], old[0], torch.tensor(.7), .2)
            loss.backward()
            outputs.append(logp.detach())
            gradients.append({n:p.grad.detach().clone() for n,p in ranker.model.named_parameters() if p.grad is not None})
            with torch.no_grad():
                greedy.append(ranker.model.generate(ids, attention_mask=mask, max_new_tokens=4, do_sample=False))
        torch.testing.assert_close(outputs[0], outputs[1], atol=2e-6, rtol=2e-6)
        assert gradients[0].keys() == gradients[1].keys()
        assert sum(float(g.norm()) for g in gradients[0].values()) > 1e-4
        for name in gradients[0]:
            torch.testing.assert_close(gradients[0][name], gradients[1][name], atol=2e-6, rtol=2e-5)
        assert torch.equal(greedy[0], greedy[1])
        assert all(p.dtype == torch.float32 for p in sdpa.model.parameters())
    finally:
        TinyModels.tearDownClass()


def test_response_projection_preserves_entire_policy_gradient():
    TinyModels.setUpClass()
    try:
        c = config('cosearch').baselines.ranker
        c.model_path = TinyModels.tmp.name
        c.attn_implementation = 'sdpa'
        c.cpu_threads = 1
        ranker = GenerativeRanker(c)
        ids = ranker.tokenizer('Question Paris Rome <ranking>[1,2,3]</ranking>', return_tensors='pt')['input_ids']
        ranker.model.eval()
        for length in (1, 4):
            records=[]
            for enabled in (False, True):
                ranker.config.response_only_logits=enabled
                ranker.optimizer.zero_grad()
                logp=ranker.response_logp(ids,length)
                loss=sequence_ppo_loss(logp[0],logp[0].detach()+.03,torch.tensor(.7),.2)
                loss.backward()
                records.append((logp.detach(),{n:p.grad.detach().clone() for n,p in ranker.model.named_parameters() if p.grad is not None}))
            torch.testing.assert_close(records[0][0],records[1][0],atol=2e-6,rtol=2e-6)
            assert records[0][1].keys()==records[1][1].keys()
            for name in records[0][1]:
                torch.testing.assert_close(records[0][1][name],records[1][1][name],atol=2e-6,rtol=2e-5)
    finally:
        TinyModels.tearDownClass()
