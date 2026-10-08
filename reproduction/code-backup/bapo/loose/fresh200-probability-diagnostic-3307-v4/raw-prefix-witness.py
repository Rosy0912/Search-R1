import sys,os,json,hashlib,importlib.util
from pathlib import Path
S=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r5-20261006');sys.path.insert(0,str(S/'repo'))
import torch
spec=importlib.util.spec_from_file_location('probe',S/'probability-diagnostic-3307-v4-20261006/tools/hf-probe.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
raw=S/'runs/formal200-6a29642035ff40b2b7787de206f886c0-3307/checkpoints/diagnostics/formal-sampling-attempt_1/formal-first-sampling-batch.pt'
if m.digest(raw)!=m.RAW_SHA:raise RuntimeError('exact raw required')
p=torch.load(raw,map_location='cpu',weights_only=False);t=p['tensors']
from verl import DataProto
from search_r1.llm_agent.sampling_tokens import audit_batch
from search_r1.baselines.fresh200_contract import critical_source_hashes
if p['source_sha256']!=critical_source_hashes():raise RuntimeError('source differs')
audit=audit_batch(DataProto.from_dict(tensors=t,non_tensors=p['non_tensor_batch'],meta_info=p['meta_info']))
def h(x):return hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest()
reports=[]
POINTS=[(1595, 575), (1865, 588), (602, 589), (286, 604), (1248, 482), (1617, 562), (433, 586), (677, 739), (2172, 662), (156, 648), (2397, 807), (500, 40), (1402, 707), (2550, 726), (988, 777), (496, 696), (630, 841), (119, 106), (1164, 569), (1713, 59), (0, 89), (1, 614)]
for row,pos in POINTS:
 turnid,off,turn=m.locate_turn(p['non_tensor_batch']['fresh200_turns'][row],pos)
 width=t['prompts'].shape[1];labelidx=width+pos
 ids=t['input_ids'][row];mask=t['attention_mask'][row].bool();positions=t['position_ids'][row]
 fullprefix=ids[:labelidx+1][mask[:labelidx+1]].tolist()
 fullpos=positions[:labelidx+1][mask[:labelidx+1]].tolist()
 request=turn['actual_request_token_ids'];generated=turn['raw_generated_token_ids'][:off+1];actualprefix=request+generated
 expectedpos=list(range(len(actualprefix)))
 ok=fullprefix==actualprefix and fullpos==expectedpos and fullprefix[-1]==int(t['responses'][row,pos]) and int(positions[labelidx-1])==len(actualprefix)-2
 if not ok:raise RuntimeError('actual prefix/position/label divergence')
 base=(row//4)*4;neighbors=[base+i for i in range(4)];changed=[row,len(t['input_ids'])//2,len(t['input_ids'])//3,len(t['input_ids'])//4];slot=neighbors.index(row);others=[x for x in changed if x!=row][:3]
 changed=others[:];changed.insert(slot,row)
 reports.append(dict(row=row,position=pos,turn=turnid,offset=off,prefix_tokens_including_label=len(actualprefix),prefix_sha256=h(actualprefix),full_prefix_sha256=h(fullprefix),position_ids_sha256=h(fullpos),prefix_and_position_and_label_exact=True,label_token=int(ids[labelidx]),full_physical_query_index=labelidx-1,full_query_position_id=int(positions[labelidx-1]),perturn_query_index=len(actualprefix)-2,full_left_padding=int((~mask[:width]).sum()),row_valid_tokens=int(mask.sum()),packed_rows=neighbors,packed_valid_tokens=[int(t['attention_mask'][i].sum()) for i in neighbors],changed_neighbor_rows=changed,changed_neighbor_valid_tokens=[int(t['attention_mask'][i].sum()) for i in changed],tested_batch_slot=slot))
print(json.dumps(dict(scope='ACTUAL_CPU_EXACT_RAW_PREFIX_POSITION_LABEL_WITNESS_NOT_NATIVE_PROBABILITY_ACCEPTANCE',source_commit=m.SOURCE,raw_sha256=m.RAW_SHA,context_audit=audit,points=reports,all22_exact=len(reports)==22 and all(x['prefix_and_position_and_label_exact'] for x in reports),optimizer_steps=0)))
