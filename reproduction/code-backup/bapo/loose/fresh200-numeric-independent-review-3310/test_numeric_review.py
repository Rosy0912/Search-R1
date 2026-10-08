import importlib.util
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('review_probe',Path(__file__).parents[1]/'fresh200-numeric-probe-3310/numeric-probe.py'); n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)

def test_flat_query_with_unequal_lengths_left_and_right_padding():
    mask=torch.tensor([[0,1,1,1,0,0],[0,0,1,1,1,1],[0,1,1,1,1,0]])
    batch={'input_ids':torch.arange(18).view(3,6),'attention_mask':mask,'responses':torch.zeros(3,3,dtype=torch.long)}
    # Width3 prompt, response offset1 means physical predictor3. Flat offset
    # excludes both preceding rows' padding and the target's left padding.
    assert n.predictor_index(batch,2,1,False)==(2,3)
    assert n.predictor_index(batch,2,1,True)==(0,9)
    actual=batch['input_ids'][mask.bool()]
    assert int(actual[9])==int(batch['input_ids'][2,3])

def test_same_logits_f64_reference_handles_large_common_offset():
    v=torch.tensor([1000.,998.,997.,990.],dtype=torch.bfloat16)
    result=n.vector_summary(v,2)
    assert result['cpu_fp64_logprob']==float(v.double()[2]-torch.logsumexp(v.double(),0))
    # BF16 quantization has already affected these logits. Casting for CE cannot
    # restore the original pre-quantization FP32 vector.
    unrounded=torch.tensor([1000.,998.,997.,990.],dtype=torch.float64)
    assert abs(result['cpu_fp64_logprob']-float(torch.log_softmax(unrounded,0)[2]))>0.1
