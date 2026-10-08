import importlib.util,json,torch
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'tree-instruct300-runtime-v3/source/repo/search_r1/baselines/gradient_norm.py';spec=importlib.util.spec_from_file_location('gradient_frozen',p);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
model=torch.nn.Linear(2,1,bias=False,dtype=torch.float64);model.weight.grad=torch.tensor([[15018.631767251734,0]],dtype=torch.float64);before=model.weight.grad.clone();reported=g.clip_grad_norm64(model,1.0);after=torch.linalg.vector_norm(model.weight.grad)
assert abs(float(reported)-15018.631767251734)<1e-9 and abs(float(after)-15018.631767251734/(15018.631767251734+1e-6))<1e-12
Path(__file__).with_name('RANK71_CLIP_CPU_SEMANTICS.json').write_text(json.dumps(dict(scope='Actual frozen clipping function on CPU non-FSDP toy, not actual3B postclip measurement',reported_preclip_norm=float(reported),postclip_norm=float(after),coefficient=float(model.weight.grad[0,0]/before[0,0]),returned_norm_is_preclip=True),indent=2)+'\n')
