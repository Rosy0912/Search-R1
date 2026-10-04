import copy
import pytest
import torch
from proof import crop_outer_padding, Tiny, loss

def fixture(real=19):
 p=torch.tensor([[0,0,4,5],[0,0,0,5]])
 r=torch.zeros(2,48,dtype=torch.long);r[:,:real]=3
 ids=torch.cat((p,r),1);a=(ids!=0).long()
 return dict(prompts=p,responses=r,input_ids=ids,attention_mask=a,position_ids=(a.cumsum(-1)-1).clamp_min(0),loss_mask=a[:,4:].clone())

def test_tail_observation_preserved():
 t=fixture();t['responses'][:,40]=7;t['input_ids'][:,44]=7;t['attention_mask'][:,44]=1
 c,s=crop_outer_padding(t,0);assert s['new_response_width']==48
 assert torch.equal(c['responses'][:,40],t['responses'][:,40]) and not c['loss_mask'][:,40].any()

def test_internal_hole_preserved():
 t=fixture();t['responses'][:,6]=0;t['input_ids'][:,10]=0;t['attention_mask'][:,10]=0;t['loss_mask'][:,6]=0
 c,_=crop_outer_padding(t,0);assert not c['attention_mask'][:,c['prompts'].shape[1]+6].any()

def test_positions_are_sliced_not_recomputed():
 t=fixture();t['position_ids']+=133
 c,_=crop_outer_padding(t,0);assert torch.equal(c['position_ids'],t['position_ids'][:,2:36])

def test_no_common_prompt_padding():
 t=fixture();t['prompts'][0,0]=8;t['input_ids'][0,0]=8;t['attention_mask'][0,0]=1
 _,s=crop_outer_padding(t,0);assert s['prompt_left_removed']==0

def test_rounding_preserves_head_chunk_widths():
 t=fixture(17);c,s=crop_outer_padding(t,0);assert s['new_response_width']==32
 assert [min(16,48-start) for start in (0,16)]==[min(16,32-start) for start in (0,16)]

@pytest.mark.parametrize('corruption',['left_id','right_id','loss_on_padding','ids_mismatch','nonbinary'])
def test_invalid_contract_rejected(corruption):
 t=fixture()
 if corruption=='left_id':t['prompts'][0,0]=8;t['input_ids'][0,0]=8
 if corruption=='right_id':t['responses'][0,47]=8;t['input_ids'][0,51]=8
 if corruption=='loss_on_padding':t['loss_mask'][0,47]=1
 if corruption=='ids_mismatch':t['input_ids'][0,3]=19
 if corruption=='nonbinary':t['attention_mask'][0,3]=2
 with pytest.raises(AssertionError):crop_outer_padding(t,0)

def test_all_inactive_response_keeps_connected_backward():
 t=fixture(0);c,s=crop_outer_padding(t,0);assert s['new_response_width']==1
 a=Tiny().double();b=copy.deepcopy(a);la=loss(a,t);lb=loss(b,c);la.backward();lb.backward()
 assert la.item()==lb.item()==0
 for x,y in zip(a.parameters(),b.parameters()):assert x.grad is not None and y.grad is not None and torch.equal(x.grad,y.grad)

def test_original_zero_width_rejected_explicitly():
 t=fixture(0);t['responses']=t['responses'][:,:0];t['input_ids']=t['prompts'];t['attention_mask']=t['attention_mask'][:,:4];t['position_ids']=t['position_ids'][:,:4];t['loss_mask']=t['loss_mask'][:,:0]
 with pytest.raises(AssertionError):crop_outer_padding(t,0)
