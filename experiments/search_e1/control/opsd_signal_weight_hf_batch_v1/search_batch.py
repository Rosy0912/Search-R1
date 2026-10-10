"""Batch only model generation; keep each frozen HF SearchAgent executor state.

HTTP requests stay one query each, so original CPU E5 request boundaries remain.
Generation callback supplies text plus actual HF IDs; all payloads stay remote.
"""
import re
from concurrent.futures import ThreadPoolExecutor

FORCED='\n<information>Search budget exhausted. Give the final answer now inside <answer> and </answer>.</information>\n'

def run_questions(questions,tokenizer,prompt,generate,http,url,max_new_tokens=500,max_context=8192,record=None):
    states=[dict(prefix=tokenizer.apply_chat_template(prompt(q),tokenize=False,add_generation_prompt=True),
        response='',search_calls=0,policy_spans=[],termination=None) for q in questions]
    for turn in range(5):
        forced=FORCED if turn==4 else ''
        active=[];texts=[]
        for i,s in enumerate(states):
            if s['termination'] is not None:continue
            text=s['prefix']+s['response']+forced
            if len(tokenizer.encode(text,add_special_tokens=False))+max_new_tokens>max_context:
                s['termination']='context_limit'
                # Native HF returns BEFORE appending forced text on this boundary.
                continue
            active.append(i);texts.append(text)
        if not active:break
        outputs=generate(texts)
        if len(outputs)!=len(active):raise ValueError('Batch decoder returned wrong count')
        pending=[]
        for i,text,output in zip(active,texts,outputs):
            s=states[i];segment=output['text']
            ends=[segment.find(tag)+len(tag) for tag in ('</search>','</answer>') if tag in segment]
            if ends:segment=segment[:min(ends)]
            s['response']+=forced;start=len(s['response']);s['response']+=segment
            s['policy_spans'].append((start,len(s['response'])))
            if record:record(dict(question_slot=i,turn=turn,input=text,generated=output,
                visible_segment=segment,policy_span=s['policy_spans'][-1]))
            if '</answer>' in segment:s['termination']='answer';continue
            match=re.search(r'<search>(.*?)</search>',segment,re.S)
            if not match or s['search_calls']==4:s['termination']='invalid_or_budget';continue
            service=http.for_case(i,turn) if hasattr(http,'for_case') else http
            pending.append((i,service,match.group(1).strip()))
        # Keep each E5 request one query; only independent requests overlap.
        # No next-turn generation until all current-turn observations are applied.
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[(i,pool.submit(service.post,url,json={'queries':[query],'topk':3,'return_scores':True},timeout=120))
                for i,service,query in pending]
            responses=[(i,f.result()) for i,f in futures]
        for i,response in responses:
            s=states[i];response.raise_for_status();passages=response.json()['result'][0]
            formatted=[]
            for n,p in enumerate(passages):
                doc=p.get('document',p);contents=doc.get('contents',doc.get('text',''))
                formatted.append(f'Doc {n+1}: {contents}')
            ids=tokenizer.encode('\n'.join(formatted),add_special_tokens=False)[:480]
            s['response']+='\n<information>\n'+tokenizer.decode(ids)+'\n</information>\n';s['search_calls']+=1
    return [{k:v for k,v in s.items() if k!='prefix'}|{'termination':s['termination'] or 'budget'} for s in states]
