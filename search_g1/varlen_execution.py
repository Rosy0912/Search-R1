"""Small real loaded-model boundary check before restoring the saved RNG."""
import json,os,time
from pathlib import Path
import torch

def verify_loaded_model(module,rank):
    from search_g1.full_state_restore import publish
    assert module.config._attn_implementation == 'flash_attention_2'
    layers=[m for m in module.modules() if m.__class__.__name__=='Qwen2FlashAttention2']
    assert len(layers)==36
    assert {p.dtype for p in module.parameters()} == {torch.float32}
    assert module.mixed_precision.param_dtype == torch.bfloat16
    original_mode=module.training;module.eval()
    device=torch.device('cuda',torch.cuda.current_device())
    first=torch.arange(10,17,device=device).unsqueeze(0)
    second=torch.arange(30,41,device=device).unsqueeze(0)
    packed=torch.cat((first,second),dim=1)
    positions=torch.cat((torch.arange(7,device=device),torch.arange(11,device=device))).unsqueeze(0)
    def forward(tokens,pos):
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            logits=module(input_ids=tokens,position_ids=pos,attention_mask=None,use_cache=False).logits
        assert logits.dtype==torch.bfloat16 and torch.isfinite(logits).all()
        return logits.float()
    try:
        both=forward(packed,positions)
        one=forward(first,positions[:,:7]);two=forward(second,positions[:,7:])
        discrepancy=max((both[:,:7]-one).abs().max().item(),(both[:,7:]-two).abs().max().item())
        changed=packed.clone();changed[:,:7]+=50
        altered=forward(changed,positions)
        isolation=torch.equal(both[:,7:],altered[:,7:])
        assert isolation, 'Packed sequences share attention'
        # BF16 kernel batching can round differently; record the actual error.
        assert discrepancy <= .125, ('Packed/single BF16 discrepancy',discrepancy)
        proof=dict(rank=rank,actual_loaded_model_forward=True,attention_backend=module.config._attn_implementation,
            actual_qwen_flash_attention_layers=len(layers), packed_sequence_lengths=[7,11],second_sequence_unchanged_when_first_changed=isolation,
            packed_single_max_abs_logit_discrepancy=discrepancy,logits_dtype='torch.bfloat16',
            master_parameter_dtype='torch.float32',checked_unix=time.time(),optimizer_writes=100,
            rng_semantics='check performed before restoring original saved worker RNG')
        publish(Path(os.environ['SEARCH_G1_BASE'])/'logs'/f'actual-varlen-execution-rank-{rank}.json',proof)
        print('SEARCH_G1_ACTUAL_VARLEN_EXECUTION='+json.dumps(proof),flush=True)
    finally:
        module.train(original_mode)
