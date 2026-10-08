import importlib.util
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('probe',Path('work/fresh200-probability-diagnostic-3307/hf-probe.py'))
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)

def test_raw_prefix_predictor_mapping_with_environment_and_left_padding():
    prompt=[10,11,12]; first=[20,21,22]; env=[30,31]; second=[40,41,42]
    turns=[dict(kept_token_count=3,raw_generated_token_ids=first,environment_token_ids=env,actual_request_token_ids=prompt),dict(kept_token_count=3,raw_generated_token_ids=second,environment_token_ids=[],actual_request_token_ids=prompt+first+env)]
    response=first+env+second; prompt_width=7; full=torch.tensor([[0]*4+prompt+response+[0]*2]); response_width=len(response)+2
    attention=torch.tensor([[0]*4+[1]*(len(prompt)+len(response))+[0]*2]);positions=(attention.cumsum(-1)-1)*attention
    prediction_indices=torch.arange(full.shape[1])[-response_width-1:-1]
    for j in range(len(response)):
        if j in (3,4):continue
        ordinal,offset,turn=probe.locate_turn(turns,j);batch=probe.turn_batch(turn)
        target=prompt_width+j;predictor=int(prediction_indices[j]);local_predictor=len(turn['actual_request_token_ids'])+offset-1
        assert int(full[0,target])==int(batch['responses'][0,offset])
        assert full[0,:target][attention[0,:target].bool()].tolist()==batch['input_ids'][0,:local_predictor+1].tolist()
        assert int(positions[0,predictor])==int(batch['position_ids'][0,local_predictor])
        assert int(batch['position_ids'][0,local_predictor])==len(prompt)+j-1
        short=probe.turn_batch(turn,offset)
        assert short['input_ids'][0,:local_predictor+1].tolist()==batch['input_ids'][0,:local_predictor+1].tolist()
        assert ordinal==(j>=5)
