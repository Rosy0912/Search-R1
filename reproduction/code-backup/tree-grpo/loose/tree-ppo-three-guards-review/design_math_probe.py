import torch,json,math
records=[]
for a in [-2.,0.,2.]:
 for d in [-100.,-1.,math.log(.8),0.,math.log(1.2),math.log(3.),2.,5.]:
  x=torch.tensor(d,dtype=torch.float64,requires_grad=True);ratio=x.exp();v=torch.maximum(-a*ratio,-a*ratio.clamp(.8,1.2));direct=torch.minimum(v,torch.tensor(-3*a,dtype=x.dtype)) if a<0 else v;g=torch.autograd.grad(direct,x)[0]
  y=torch.tensor(d,dtype=torch.float64,requires_grad=True)
  if a<0: bounded=-a*y.clamp(math.log(.8),math.log(3)).exp()
  elif a>0:bounded=-a*y.clamp(max=math.log(1.2)).exp()
  else:bounded=y*0
  h=torch.autograd.grad(bounded,y)[0]
  assert torch.allclose(direct,bounded,atol=1e-12,rtol=1e-12)
  if d not in [math.log(.8),math.log(1.2),math.log(3.)]:assert torch.allclose(g,h,atol=1e-12,rtol=1e-12)
  records.append(dict(advantage=a,delta=d,loss=direct.item(),direct_grad=g.item(),bounded_grad=h.item(),boundary=d in [math.log(.8),math.log(1.2),math.log(3.)]))
print(json.dumps(dict(status='CPU_FLOAT64_VALUE_EQUIVALENCE_NONBOUNDARY_GRADIENT_PASS',torch_version=torch.__version__,records=records)))
