import json,pathlib,socket,hashlib,os
from .protocol import peer,send,receive

class Client:
    def __init__(self,manifest_path,teacher_path,student_tokenizer):
        self.manifest_path=pathlib.Path(manifest_path);self.config=json.loads(self.manifest_path.read_text());self.tokenizer=student_tokenizer
        if pathlib.Path(teacher_path).resolve()!=pathlib.Path(self.config['teacher_path']).resolve():raise ValueError('Trained teacher path mismatch')
        token_sha=hashlib.sha256(json.dumps(student_tokenizer.get_vocab(),sort_keys=True).encode()).hexdigest()
        if token_sha!=self.config['token_vocab_sha256']:raise ValueError('Complete token-to-ID vocabulary mismatch')
        self.scored_tokens=0;self.parameter_count=32763876352;self.last_release=None;self.last_receipts=[]
    def score(self,prompts,responses,attention,temperature=1.,chunk=16,outer_step=None):
        import torch
        from safetensors.torch import save,load
        if temperature!=1. or outer_step is None:raise ValueError('TOPD teacher temperature/order')
        tensors={k:t.detach().long().cpu().contiguous() for k,t in [('prompts',prompts),('responses',responses),('attention',attention)]};payload=save(tensors)
        with socket.socket(socket.AF_UNIX) as connection:
            connection.settimeout(14400);connection.connect(self.config['socket']);actual=peer(connection,self.config['job'])
            if actual['pid']!=self.config['server_pid'] or actual['starttime']!=self.config['server_starttime']:raise ValueError('Unknown teacher PID')
            send(connection,dict(capability=self.config['capability'],outer_step=outer_step,teacher_path=self.config['teacher_path'],temperature=temperature,targets=None,student_pad_token_id=self.tokenizer.pad_token_id),payload)
            meta,body=receive(connection)
            if meta['status']!='ACTUAL_TP8_KV_RELEASE_ACK_BEFORE_STUDENT_UPDATE' or meta['outer_step']!=outer_step or meta['job']!=self.config['job'] or meta['request_sha256']!=hashlib.sha256(payload).hexdigest() or meta['output_sha256']!=hashlib.sha256(body).hexdigest():raise ValueError('Actual teacher receipt identity mismatch')
            ack=meta['released_workers']
            if len(ack)!=8 or {r['rank'] for r in ack}!=set(range(8)) or any(not r['cache_engine_absent'] or not r['gpu_cache_absent'] or r['requires_grad_parameters'] for r in ack):raise ValueError('Actual TP8 release ACK incomplete')
            values=load(body)['teacher_log_probs']
            if values.shape!=responses.shape or not bool(torch.isfinite(values).all()) or not bool((values[tensors['attention']==0]==0).all()):raise ValueError('Teacher output/mask/finite mismatch')
            send(connection,dict(status='CLIENT_ACCEPTED_RELEASED_RESULT',outer_step=outer_step,output_sha256=meta['output_sha256']))
        self.scored_tokens+=meta['scored_tokens'];self.last_release=meta;self.last_receipts=[meta];return values
    def require_released(self,outer_step):
        if self.last_release is None or self.last_release['outer_step']!=outer_step:raise ValueError('Teacher result not released before actor update')
