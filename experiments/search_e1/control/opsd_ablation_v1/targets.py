"""One shared base; frozen teacher/outer-old; same causal D continuation."""
from contextlib import contextmanager
import hashlib
import json


def tensor_digest(values):
    import torch
    h=hashlib.sha256()
    for name,t in sorted(values.items()):
        t=t.detach().cpu().contiguous()
        h.update(json.dumps([name,str(t.dtype),list(t.shape)],separators=(',',':')).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def causal_hidden(hidden,prefix_length,response_length):
    if prefix_length<1 or response_length<1 or hidden.shape[0]!=prefix_length+response_length:
        raise ValueError('Causal input shape/prefix mismatch')
    return hidden[prefix_length-1:prefix_length+response_length-1]


class SharedTargets:
    def __init__(self,model,precision,current_adapter='default',old_adapter=None,old_identity=None):
        self.model,self.precision=model,precision
        self.current_adapter,self.old_adapter=current_adapter,old_adapter
        self.old_identity=None if old_identity is None else json.loads(json.dumps(old_identity));self.busy=False
        if old_adapter and (old_adapter==current_adapter or not isinstance(old_identity,dict) or not old_identity.get('adapter_tensor_sha256')):
            raise ValueError('Distinct frozen outer-start adapter with explicit identity required')
        self.old_parameters=[(n,p) for n,p in model.named_parameters() if old_adapter and '.'+old_adapter+'.' in n]
        if old_adapter and not self.old_parameters:raise ValueError('Missing outer-old adapter')
        if old_adapter:
            actual=tensor_digest({n.replace('.'+old_adapter+'.','.'):p for n,p in self.old_parameters})
            if actual!=old_identity['adapter_tensor_sha256']:raise ValueError('Outer-old actual tensors differ from bound start checkpoint')
        self.old_versions={n:p._version for n,p in self.old_parameters}
        for _,p in self.old_parameters:p.requires_grad_(False)
        self.activate_current()

    def activate_current(self):
        self.model.set_adapter(self.current_adapter)
        for n,p in self.model.named_parameters():
            p.requires_grad_('lora_' in n and '.'+self.current_adapter+'.' in n)

    def check_old(self):
        if any(p._version!=self.old_versions[n] or p.requires_grad for n,p in self.old_parameters):
            raise ValueError('Frozen outer-old adapter changed')

    @contextmanager
    def current_graph(self):
        if self.busy:raise ValueError('Overlapping current graph/adapter switching forbidden')
        self.activate_current();self.check_old();self.busy=True
        self.current_versions={n:(p._version,p.requires_grad) for n,p in self.model.named_parameters()}
        try:yield self
        finally:
            active=self.model.active_adapter
            if active!=self.current_adapter:raise ValueError('Adapter changed before current backward completed')
            if any((p._version,p.requires_grad)!=self.current_versions[n] for n,p in self.model.named_parameters()):
                raise ValueError('Parameters/gradient flags changed during current graph')
            self.check_old();self.busy=False;self.precision.stage='forward'

    def mark_backward(self):
        if not self.busy:raise ValueError('Backward outside current graph lease')
        self.precision.stage='backward_recompute'
        self.precision.role='current'

    def hidden(self,encoded):
        import torch
        if not self.busy:raise ValueError('Hold current_graph through backward')
        prefixes,response=encoded['prefixes'],encoded['response']
        base=self.model.get_base_model();transformer=base.base_model
        device=next(self.model.parameters()).device
        def forward(prefix,role):
            ids=torch.tensor([prefix+response],device=device)
            with self.precision.context(role):result=transformer(input_ids=ids,use_cache=False).last_hidden_state[0]
            if not torch.isfinite(result).all().item():raise FloatingPointError('Nonfinite '+role+' forward activation')
            return causal_hidden(result,len(prefix),len(response))
        self.model.eval()
        old=None
        try:
            with torch.no_grad(),self.model.disable_adapter():teacher=forward(prefixes[1],'teacher').detach()
            if self.old_adapter:
                self.model.set_adapter(self.old_adapter)
                for _,p in self.old_parameters:p.requires_grad_(False)
                with torch.no_grad():old=forward(prefixes[0],'outer_old').detach()
        finally:
            self.activate_current();self.check_old()
            # Restore current before propagating target-forward failure too.
            self.current_versions={n:(p._version,p.requires_grad) for n,p in self.model.named_parameters()}
        self.model.train()
        current=forward(prefixes[0],'current')
        return current,teacher,old
