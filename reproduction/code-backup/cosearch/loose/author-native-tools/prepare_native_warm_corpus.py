"""Approved additional SAME-base3B native-grammar SFT, training labels only; 0 RL."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,sys
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer
root=Path('/dataset/shaoshitong/search-r1-rl/cosearch');native=root/'author-native'
sys.path[:0]=[str(native/'repo'),str(native/'repo/verl')]
from verl.tools.utils.prompts import SEARCH_R1_CoT_PROMPT,RERANK_PROMPT_WITH_INITIAL_QUERY
from verl.tools.utils.validates import validate_rerank_output
from verl.experimental.agent_loop.tool_parser import SearchR1ToolParser
from verl.experimental.agent_loop.co_search_agent_loop import CoSearchAgentLoop
import asyncio
from types import SimpleNamespace
from verl.tools.utils.search import format_tool_response,format_tool_response_with_docid_map
n=root/'runs/cosearch-local-gpu-five-20261003-n/warm-start'
source=n/'training-only-corpus.json';manifest=json.loads((n/'manifest.json').read_text())
assert manifest['completed'] and manifest['stage_periodic_and_final_questions_excluded']
assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['corpus_sha256']
rows=json.loads(source.read_text());assert len(rows)==128
train=pd.read_parquet(native/'data/train.parquet');valid=pd.read_parquet(native/'data/validation-64.parquet')
questions={str(e['question']) for e in train.extra_info};excluded={str(e['question']) for e in valid.extra_info}
tok=AutoTokenizer.from_pretrained(root/'assets-root/assets/qwen2.5-3b',local_files_only=True)
actors=[];rankers=[];records=[]
def example(messages,target):
 prefix=tok.apply_chat_template(messages,add_generation_prompt=True,tokenize=True)
 answer=tok.encode(target,add_special_tokens=False)+[tok.eos_token_id]
 return dict(input_ids=prefix+answer,response_length=len(answer),prompt_tokens=len(prefix),target=target)
for r in rows:
 question=r['question'];assert question in questions and question not in excluded
 raw_docs=r['candidates'];assert len(raw_docs)==50
 docs=[dict(d['document'],score=d.get('score')) for d in raw_docs]
 assert all(d.get('contents') for d in docs)
 ranking=r['supervised_ranking'];assert len(ranking)==5 and len(set(ranking))==5 and all(1<=v<=50 for v in ranking)
 prompt=SEARCH_R1_CoT_PROMPT.format(question=question)
 call={'name':'search','arguments':{'query':question}}
 action='<reason>I need evidence to answer this question.</reason>\n<tool_call>'+json.dumps(call,ensure_ascii=False)+'</tool_call>'
 assert json.loads(action.split('<tool_call>')[1].split('</tool_call>')[0])==call
 user={'role':'user','content':prompt}
 actors.append(example([user],action))
 # Explicit full-document adaptation for SFT, no default2000-character crop.
 max_chars=max(len(d['contents']) for d in docs)+1
 observation=format_tool_response([docs[i-1] for i in ranking],max_doc_length=max_chars)
 final='<reason>I will provide a concise answer.</reason>\n<answer>'+r['aliases'][0]+'</answer>'
 actors.append(example([user,{'role':'assistant','content':action},{'role':'tool','content':observation}],final))
 passages,mapping=format_tool_response_with_docid_map(docs,max_doc_length=max_chars)
 assert all(d['contents'] in passages for d in docs), 'No full candidate text may be lost'
 rank_prompt=RERANK_PROMPT_WITH_INITIAL_QUERY.format(N=50,M=5,initial_query=question,sub_query=question,passages_block=passages)
 target='<reason>I rank passages by relevance to the question.</reason>\n<rerank>'+' > '.join('['+str(i)+']' for i in ranking)+'</rerank>'
 rankers.append(example([{'role':'user','content':rank_prompt}],target))
 assert validate_rerank_output(target,N=50,M=5,docid_map=mapping)['error_type'] is None
 assert CoSearchAgentLoop.detect_answer(SimpleNamespace(tokenizer=tok),tok.encode(final,add_special_tokens=False))
 _,calls=asyncio.run(SearchR1ToolParser(tok).extract_tool_calls(tok.encode(action,add_special_tokens=False)))
 assert len(calls)==1 and calls[0].name=='search' and json.loads(calls[0].arguments)['query']==question
 records.append(dict(question=question,source_index=r['source_index'],candidates=docs,docid_map=mapping,ranking=ranking,ranker_prompt_tokens=rankers[-1]['prompt_tokens']))
out=native/'native-grammar-warm-start-fulltext-v2';out.mkdir(exist_ok=True)
corpus=dict(actor=actors,ranker=rankers,records=records)
p=out/'corpus.json';assert not p.exists(),'Immutable separate native corpus';p.write_text(json.dumps(corpus,ensure_ascii=False)+'\n')
report=dict(prepared=True,real_rl_updates=0,initial_model=str(root/'assets-root/assets/qwen2.5-3b'),reuse_N_warm_weights=False,questions=128,actor_examples=len(actors),ranker_examples=len(rankers),proposed_epochs=1,proposed_batch_size=8,proposed_actor_sft_steps=32,proposed_ranker_sft_steps=16,source_training_corpus=str(source),source_training_corpus_sha256=manifest['corpus_sha256'],source_native_train_sha256=hashlib.sha256((native/'data/train.parquet').read_bytes()).hexdigest(),corpus_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),all_questions_retained_train=True,validation64_excluded=True,stage_and_final_exclusions_inherited_from_source_manifest=True,gold_only_in_supervised_targets_and_training_ranking=True,full50_docs=True,original_native_all_target_validators_passed=True,document_character_crop=False,ranker_prompt_token_minmax=[min(v['prompt_tokens'] for v in rankers),max(v['prompt_tokens'] for v in rankers)],actor_prompt_token_minmax=[min(v['prompt_tokens'] for v in actors),max(v['prompt_tokens'] for v in actors)],grammar='original native reason/tool_call search JSON + answer; reason/rerank bracket-chain',training_not_started=True,full_context_adaptation_requires_separate_review=True)
(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
