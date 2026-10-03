"""Batched fresh rollouts; no changes to the GRPO stage."""
import re
import requests
from vllm import LLM, SamplingParams
from .core import prompt

def batched_rollouts(model, rows, k, url, temperature=1., seed=42, batch_size=64):
    llm = LLM(model=model, tensor_parallel_size=1, dtype='bfloat16', gpu_memory_utilization=0.6,
              max_model_len=8192, enforce_eager=True, seed=seed)
    tokenizer = llm.get_tokenizer()
    for start in range(0,len(rows),batch_size):
        states=[]
        for row in rows[start:start+batch_size]:
            prefix=tokenizer.apply_chat_template(prompt(row['question']),tokenize=False,add_generation_prompt=True)
            for sibling in range(k):
                states.append(dict(row,prefix=prefix,response='',search_calls=0,policy_spans=[],sibling=sibling,termination=None))
        for turn in range(5):
            active=[]
            for state in states:
                if state['termination'] is not None:
                    continue
                if turn == 4:
                    state['response'] += '\n<information>Search budget exhausted. Give the final answer now inside <answer> and </answer>.</information>\n'
                text=state['prefix']+state['response']
                if len(tokenizer.encode(text,add_special_tokens=False))+500>8192:
                    state['termination']='context_limit'
                else:
                    active.append(state)
            if not active:
                break
            params=SamplingParams(temperature=temperature,top_p=1.,max_tokens=500,
                                  stop=['</search>','</answer>'],include_stop_str_in_output=True)
            outputs=llm.generate([s['prefix']+s['response'] for s in active],params,use_tqdm=False)
            searches=[]
            for state,output in zip(active,outputs):
                segment=output.outputs[0].text
                begin = len(state['response'])
                state['response']+=segment
                state['policy_spans'].append((begin,len(state['response'])))
                if '</answer>' in segment:
                    state['termination']='answer'
                else:
                    match=re.search(r'<search>(.*?)</search>',segment,re.S)
                    if match and state['search_calls']<4:
                        searches.append((state,match.group(1).strip()))
                    else:
                        state['termination']='invalid_or_budget'
            if searches:
                r=requests.post(url,json={'queries':[q for _,q in searches],'topk':3,'return_scores':True},timeout=300)
                r.raise_for_status()
                docs=r.json()['result']
                if len(docs)!=len(searches):
                    raise RuntimeError('Retriever returned wrong batch size')
                for (state,_),passages in zip(searches,docs):
                    contents='\n'.join(f"Doc {i+1}: {p.get('document',p).get('contents','')}" for i,p in enumerate(passages))
                    info=tokenizer.decode(tokenizer.encode(contents,add_special_tokens=False)[:480])
                    state['response']+='\n<information>\n'+info+'\n</information>\n'
                    state['search_calls']+=1
        for state in states:
            state['token_count'] = len(tokenizer.encode(state['response'], add_special_tokens=False))
            state.pop('prefix')
            state['termination']=state['termination'] or 'budget'
            yield state
