"""Dependency-free protocol, scorer and immutable-identity checks."""
import hashlib
import json
import re
import string
from collections import Counter

PROFILE = 'qwen25_7b_common_eval_v2_20261019'
COUNTS = dict(nq=3610, triviaqa=11313, popqa=14267, hotpotqa=7405,
              **{'2wikimultihopqa':12576}, musique=2417, bamboogle=125)
TOTAL = 51713
IDS_SHA = '4e9df91a75beba40c3316576f6da049851e736df4a1e5c873f4f0cdb883fd4b1'
PREFIX = ('Answer the given question. '
 'You must conduct reasoning inside <think> and </think> first every time you get new information. '
 'After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. '
 'You can search as many times as your want. '
 'If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: ')
SCIENCE = dict(normal_interactions=4, final_generation_for_active=True, max_prompt_tokens=4096,
 max_start_tokens=2048, max_new_tokens=500, max_observation_tokens=500, topk=3,
 temperature=0.0, top_p=1.0, top_k=-1, engine_seed=0, n=1, dtype='bfloat16',
 native_stop=['</search>'], stop_token_ids=[151645,151643], repetition_penalty=1.0,
 presence_penalty=0.0, frequency_penalty=0.0, answer_rule='last_answer_tag_in_model_authored_spans',
 question_prompt='official_SearchR1_598e61b_make_prefix', retain_old_results=True)

def digest(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def prompt(question):
 q=question.strip()
 if not q:raise ValueError('Empty question')
 if q[-1]!='?':q+='?'
 return PREFIX+q+'\n'

def question_from_old_prompt(chat):
 if len(chat)!=1 or chat[0]['role']!='user':raise ValueError('Unexpected common prompt structure')
 text=chat[0]['content']
 marker='\nQuestion: '
 if text.count(marker)!=1:raise ValueError('Ambiguous original question delimiter')
 return text.split(marker,1)[1].strip()

def normalize(text):
 text=''.join(c for c in text.lower() if c not in string.punctuation)
 return ' '.join(re.sub(r'\b(a|an|the)\b',' ',text).split())

def answer(spans):
 # The official extractor's >1 test accounts for the single prompt example.
 # Restrict the search area to model-authored text, excluding retrieved documents.
 matches=list(re.finditer(r'<answer>(.*?)</answer>','\n'.join(spans),re.DOTALL))
 return matches[-1].group(1).strip() if matches else None

def em(pred,gold):
 return int(pred is not None and any(normalize(pred)==normalize(g) for g in gold))

def strict_format(spans):
 if not spans:return False
 text=spans[-1].strip();m=re.search(r'<answer>(.*?)</answer>',text,re.DOTALL)
 if not m or text[m.end():].strip() or re.search(r'</?(?:think|search|answer|information)\b',m.group(1)):return False
 prefix=text[:m.start()].strip()
 if not prefix:return True
 thought=re.fullmatch(r'<think>(.*?)</think>',prefix,re.DOTALL)
 return bool(thought and not re.search(r'</?(?:think|search|answer|information)\b',thought.group(1)))

def action(text):
 m=re.search(r'<(search|answer)>(.*?)</\1>',text,re.DOTALL)
 return (m.group(1),m.group(2).strip()) if m else (None,'')

def postprocess(text):
 if '</search>' in text:return text.split('</search>',1)[0]+'</search>'
 if '</answer>' in text:return text.split('</answer>',1)[0]+'</answer>'
 return text

def passages(results):
 out=''
 for i,item in enumerate(results):
  c=item['document']['contents'];title=c.split('\n')[0];body='\n'.join(c.split('\n')[1:])
  out+=f'Doc {i+1}(Title: {title}) {body}\n'
 return out

INVALID='\nMy previous action is invalid. If I want to search, I should put the query between <search> and </search>. If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n'

def validate_rows(rows):
 if len(rows)!=TOTAL or Counter(r['source'] for r in rows)!=Counter(COUNTS):raise ValueError('Full fixed seven datasets required')
 ids=[r['question_id'] for r in rows]
 if len(set(ids))!=TOTAL or any(r['final_index']!=i or not r['gold_answers'] for i,r in enumerate(rows)):raise ValueError('Question identity/order/gold mismatch')

def metrics(rows):
 groups={k:[r for r in rows if r['source']==k] for k in COUNTS};groups['all']=rows
 result={}
 for k,g in groups.items():
  n=len(g);result[k]=dict(count=n,correct=sum(r['em'] for r in g),
   em_percent=100*sum(r['em'] for r in g)/n,
   mean_generated_tokens=sum(r['generated_tokens'] for r in g)/n,
   mean_search_calls=sum(r['search_calls'] for r in g)/n,
   mean_raw_emitted_tokens=sum(r['raw_emitted_tokens'] for r in g)/n,
   strict_format_rate=sum(r['format_valid'] for r in g)/n)
 result['equal_dataset']={key:sum(result[k][key] for k in COUNTS)/7 for key in
  ['em_percent','mean_generated_tokens','mean_search_calls','mean_raw_emitted_tokens']}
 return result
