import datetime
import hashlib
import json
import os
import re
import subprocess

BASE = '/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-fresh200-train-3326'
ROOT = os.path.join(BASE, 'checkpoints')
BLOCKED = ('question', 'context', 'token', 'logprob', 'log_prob', 'logps', 'retrieval', 'prompt', 'response', 'weight', 'index', 'local_indices', 'ledger')
STRINGS = {'state', 'status', 'stage', 'decision', 'continuation_decision', 'reason', 'job_id', 'run_id', 'phase', 'scheduler_type', 'optimizer_type', 'code_commit', 'source_commit', 'scope', 'schema', 'schema_version', 'kind', 'outcome', 'checkpoint_loaded', 'checkpoint_type', 'checkpoint_format', 'native_resume_supported', 'resume_supported'}
def selected(value, key=''):
    if any(term in key.lower() for term in BLOCKED):
        return None
    if isinstance(value, dict):
        return {k: selected(v,k) for k,v in value.items() if not any(term in k.lower() for term in BLOCKED) and (isinstance(v,(dict,bool,int,float)) or v is None or isinstance(v,str) and (k in STRINGS or k.endswith(('_sha256','_sha')) or k == 'sha256'))}
    if isinstance(value,(bool,int,float)) or value is None:
        return value
    if isinstance(value,str) and (key in STRINGS or key.endswith(('_sha256','_sha')) or key == 'sha256'):
        return value[:250]
    return None
result={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':3326,'checkpoint_root_exists':os.path.isdir(ROOT),'directories':[],'files':[],'json_manifests':[],'fatal_error_metadata':[]}
for directory, children, names in os.walk(ROOT):
    children.sort()
    result['directories'].append(os.path.relpath(directory,BASE))
    for name in sorted(names):
        path=os.path.join(directory,name)
        if not os.path.isfile(path):
            continue
        stat=os.stat(path)
        record={'name':os.path.relpath(path,BASE),'bytes':stat.st_size,'mtime':stat.st_mtime,'sha256':None}
        if stat.st_size<=1024*1024:
            with open(path,'rb') as f:
                data=f.read()
            record['sha256']=hashlib.sha256(data).hexdigest()
            if name.endswith('.json') and ('manifest' in name or 'state' in name or 'attempt-' in name):
                try:
                    obj=json.loads(data)
                    result['json_manifests'].append({'name':record['name'],'metadata':selected(obj)})
                except (json.JSONDecodeError,UnicodeDecodeError):
                    record['json_valid']=False
        else:
            record['sha256_not_computed_in_initial_inventory']=True
        result['files'].append(record)
stage=os.path.join(BASE,'stage.log')
with open(stage,errors='replace') as f:
    for line in f:
        line=re.sub(r'\x1b\[[0-9;]*m','',line)
        match=re.search(r'\b(ValueError|RuntimeError|UnicodeDecodeError):\s*(.*)',line)
        if match:
            message=match.group(2).strip()
            # Fixed runtime error text is returned only when it has no user-content carriers.
            record={'exception_class':match.group(1),'message_chars':len(message),'message_sha256':hashlib.sha256(message.encode()).hexdigest()}
            if not re.search(r'<(?:/?search|/?answer|/?information)\b|\b(?:question|context|prompt|token_ids|input_ids|log_probs|logps)\s*[:=]|\[|\]|\{|\}|\b(?:tensor|array)\s*\(|["\']',message,re.I) and len(message)<=240:
                record['fixed_error_message']=message
            result['fatal_error_metadata'].append(record)
for name in ['run-result.json','output/training-rollout-cost/attempt-000002.json']:
    path=os.path.join(BASE,name)
    with open(path) as f:
        obj=json.load(f)
    error=obj.get('error')
    if isinstance(error,str):
        result['fatal_error_metadata'].append({'origin':name,'error_type':obj.get('error_type'),'message_chars':len(error),'message_sha256':hashlib.sha256(error.encode()).hexdigest()})
control=subprocess.run(['scontrol','show','job','3326'],capture_output=True,text=True)
queue=subprocess.run(['squeue','-h','-j','3326','-o','%i %T %R'],capture_output=True,text=True)
result['scontrol_positive_terminal_text']=control.stdout if control.returncode==0 else None
result['scontrol_returncode']=control.returncode
result['squeue_own_text']=queue.stdout
result['squeue_returncode']=queue.returncode
print(json.dumps(result,sort_keys=True))
