from pathlib import Path
import subprocess
r=Path('work/bapo-numerics');commit='4dc50b033929fc26715fee3526bffbf6055254d5'
for name in ['search_r1/baselines/gradient_norm.py','tests/baselines/test_gradient_numerics.py']:
 p=r/name;assert not p.exists();p.write_bytes(subprocess.check_output(['git','-C',str(r),'show',commit+':'+name]))
p=r/'verl/trainer/ppo/core_algos.py';s=p.read_text();old='''        kl = ref_logprob - logprob
        ratio = torch.exp(kl)''';new='''        kl = ref_logprob - logprob
        if not torch.isfinite(kl).all():
            raise ValueError('Nonfinite low_var_kl log probability difference')
        # Outside [-11,4], exp(x)-x-1 is already above the existing
        # upper clamp10, with zero derivative. Bound the input before exp
        # to preserve that function without the 0*inf backward NaN.
        kl = torch.clamp(kl, min=-11., max=4.)
        ratio = torch.exp(kl)''';assert s.count(old)==1;p.write_text(s.replace(old,new))
p=r/'verl/workers/actor/dp_actor.py';s=p.read_text();old='''        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)''';new='''        from search_r1.baselines.gradient_norm import clip_grad_norm64
        grad_norm = clip_grad_norm64(self.actor_module, self.config.grad_clip)''';assert s.count(old)==1;p.write_text(s.replace(old,new))
