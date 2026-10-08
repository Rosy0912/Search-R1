"""Synthetic parser/identity guards. No actual trajectories or RL acceptance."""
import copy, json
from check_native_five import semantic_groups

class Tokenizer:
    def decode(self,ids,**kwargs):
        order=[1,2,3,4,5] if ids==[1] else [5,4,3,2,1]
        return '<reason>synthetic CPU diagnostic</reason><rerank>'+' > '.join(f'[{i}]' for i in order)+'</rerank>'

row=dict(uids=['group']*2,rewards=[1.,0.],score_sums={'token_level_scores':[1.,0.]},
    advantages=[[1.],[-1.]],response_ids=[[1],[2]],response_masks=[[1],[1]],
    reward_metadata={'tool_score':[1.,1.],'sub_query':['query','query']})
from verl.tools.utils.search import format_tool_response_with_docid_map
from verl.tools.utils.validates import validate_rerank_output
documents=[dict(id=f'doc-{i}',contents=f'passage {i}',score=.5,title='') for i in range(50)]
passages,mapping=format_tool_response_with_docid_map(documents)
validated=validate_rerank_output(Tokenizer().decode([1]),50,5,mapping)
assert validated['error_type'] is None
assert validated['reranked_docs']==documents[:5]
inventory=[document['id'] for document in documents]
assert [document['id'] for document in validated['reranked_docs']]==inventory[:5]
long_docs=[dict(id='long',contents='x'*2001,score=.1)]
cropped,_=format_tool_response_with_docid_map(long_docs)
assert 'x'*2001 not in cropped and 'x'*2000+'...' in cropped
full,_=format_tool_response_with_docid_map(long_docs,max_doc_length=2001)
assert 'x'*2001 in full
first=dict(query='query',response_ids=[1],candidate_count=50,top_m=5,
    original_error_type=None,actual_candidate_ids=inventory,actual_reranked_docs=inventory[:5])
second=dict(first,response_ids=[2])
assert semantic_groups(row,Tokenizer(),[first,second])==0,'Index/text differences do not prove different document rankings'
second=dict(second,actual_reranked_docs=list(reversed(inventory[:5])))
assert semantic_groups(row,Tokenizer(),[first,second])==1
constant=copy.deepcopy(row);constant['score_sums']['token_level_scores']=[1.,1.]
assert semantic_groups(constant,Tokenizer(),[first,second])==0
fallback=copy.deepcopy(row);fallback['reward_metadata']['tool_score'][0]=-.2
assert semantic_groups(fallback,Tokenizer(),[first,second])==0
malformed=copy.deepcopy(row);malformed['response_masks'][1]=[]
try:semantic_groups(malformed,Tokenizer(),[first,second])
except AssertionError:pass
else:raise AssertionError('Malformed arrays must fail closed')
print(json.dumps(dict(synthetic_cpu_only=True,real_rl_updates=0,physical_doc_identity_checked=True,actual_author_flat_document_schema=True,author_character_crop_verified=True,
    fallback_and_constant_rewards_excluded=True,malformed_arrays_rejected=True,passed=True)))
