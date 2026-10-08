"""Non-author analytic gradient audit of captured native-actor records."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def reference(rows_total, selected, empty_rank, pg, kl, all_empty):
 i=np.arange(rows_total,dtype=np.float64)[:,None];t=np.arange(7,dtype=np.float64)[None,:]
 f=np.stack([.4*np.sin(.17*i+.31*t+.13*k)for k in range(16)],axis=-1)
 valid=t<(i.astype(np.int64)%8)
 if empty_rank:valid[:rows_total//8]=False
 if all_empty:valid[:]=False
 w=np.linspace(-.04,.05,16);now=-3+f@w
 old=-3+.35*np.cos(.21*i+.5*t);ref=-3+.8*np.cos(.1*i+.3*t)
 adv=2*np.sin(.27*i+.4*t) if pg else np.zeros_like(now)
 ratio=np.exp(now-old);raw=-adv*ratio;clipped=-adv*np.clip(ratio,.8,1.2)
 # max clipped/raw; saturated clip branch has zero derivative.
 dpg=np.where(raw>=clipped,-adv*ratio,0.)
 x=ref-now;pre=np.exp(x)-x-1
 dkl=np.where((pre>-10)&(pre<10),1-np.exp(x),0.) if kl else np.zeros_like(now)
 derivative=np.where(valid,dpg+.001*dkl,0.)/np.maximum(valid.sum(-1,keepdims=True),1)
 ix=np.asarray(selected);return (f[ix]*derivative[ix,:,None]).sum(axis=(0,1))/len(ix)

def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();evidence=[];maximum=0.;count=0
 for rank in range(8):
  q=a.run/'native-actor'/f'rank-{rank}.json';v=json.loads(q.read_text());assert v['rank']==rank and len(v['records'])==62
  for r in v['records']:
   n=r['global_rows'];local=n//8;steps=local//16;assert len(r['gradients'])==steps
   for s,g in enumerate(r['gradients']):
    ids=[j*local+k for j in range(8)for k in range(s*16,(s+1)*16)]
    want=reference(n,ids,r.get('empty_rank',False),r['pg_on'],r['kl_on'],r['case']=='actual_actor_global_empty_step')
    got=np.asarray(g,dtype=np.float64);err=np.abs(got-want);atol=2e-4 if r['production_bf16_fp32'] else 2e-7;rtol=.03 if r['production_bf16_fp32'] else 2e-5
    assert np.all(err<=atol+rtol*np.abs(want)),(rank,r['case'],r['micro_per_rank'],s,err.max())
    maximum=max(maximum,float(err.max()));count+=1
  evidence.append({'path':str(q),'sha256':hashlib.sha256(q.read_bytes()).hexdigest()})
 out={'scope':'INDEPENDENT_ANALYTIC_GRADIENT_AUDIT_ONLY_NOT_HARDWARE_RECEIPT','rank_files':8,'records':496,'gradient_vectors_checked':count,'coordinates_per_vector':16,'maximum_absolute_gradient_error':maximum,'formula':'analytical FP64 derivative of sequence-mean clipped PPO plus .001 saturated KL, whole global mini denominator','evidence':evidence}
 with a.output.open('x')as f:json.dump(out,f,indent=2)
 print(json.dumps(out))
if __name__=='__main__':main()
