"""Conditional G1 deployment contracts; reward formulas live in main_ppo_eco."""
import hashlib, json, math, pathlib, subprocess

def validate_batch_layout(mini,micro,world,n=1):
    if any(int(v)!=v or v<=0 for v in (mini,micro,world,n)): raise ValueError('positive integer layout required')
    if mini%world or micro%world: raise ValueError('global batches must divide world exactly')
    local_mini=mini//world*n; local_micro=micro//world*n
    if local_mini%local_micro: raise ValueError('local mini must divide local micro exactly')
    return local_mini,local_micro

def probe_payload(items,manifest_path):
    raw=pathlib.Path(manifest_path).read_bytes(); manifest=json.loads(raw)
    identity={'manifest_sha256':hashlib.sha256(raw).hexdigest(),'encoder':manifest['encoder'],'round':manifest['round']}
    rows=[]
    for i,item in enumerate(items):
        row=dict(item); row['row_id']=hashlib.sha256(json.dumps([i,item['prompt'],item['response']],ensure_ascii=False).encode()).hexdigest(); rows.append(row)
    return {'items':rows,'identity':identity}

def validate_probe_response(response,payload):
    if response.get('identity')!=payload['identity']: raise ValueError('probe frozen encoder/manifest identity mismatch')
    rows=response.get('results')
    if not isinstance(rows,list) or len(rows)!=len(payload['items']): raise ValueError('probe row count mismatch')
    for expected,row in zip(payload['items'],rows):
        if row.get('row_id')!=expected['row_id']: raise ValueError('probe row identity/order mismatch')
        for name in ('flip','boundary'):
            value=row.get(name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1: raise ValueError('invalid probe probability')
    return rows

def refresh_at_outer_barrier(command,checkpoint,step):
    """Synchronous argv only; training cannot advance before success."""
    if not isinstance(command,(list,tuple)) or not command or not all(isinstance(x,str) and x for x in command): raise ValueError('refresh command must be argv')
    subprocess.run(list(command)+['--checkpoint',str(checkpoint),'--outer-step',str(step)],check=True)

from contextlib import contextmanager
@contextmanager
def probe_training_phase(probe_url,manifest_path,session=None):
    """Optional owned loopback encoder unload around actual actor update."""
    if not probe_url:
        yield
        return
    if not probe_url.startswith('http://127.0.0.1:'):raise ValueError('probe phase requires own loopback URL')
    import requests
    session=session or requests.Session();session.trust_env=False
    raw=pathlib.Path(manifest_path).read_bytes();manifest=json.loads(raw)
    expected={'round':manifest['round'],'encoder':manifest['encoder'],'manifest_sha256':hashlib.sha256(raw).hexdigest()}
    base=probe_url.rsplit('/',1)[0]
    response=session.post(base+'/barrier_pause',timeout=600);response.raise_for_status()
    if response.json().get('status')!='PAUSED_ENCODER_RELEASED':raise ValueError('unknown probe phase pause ACK')
    try:yield
    finally:
        response=session.post(base+'/barrier_resume',timeout=1200);response.raise_for_status();receipt=response.json()
        if receipt.get('status')!='READY' or any(receipt.get(k)!=v for k,v in expected.items()):raise ValueError('probe phase resume encoder changed')
