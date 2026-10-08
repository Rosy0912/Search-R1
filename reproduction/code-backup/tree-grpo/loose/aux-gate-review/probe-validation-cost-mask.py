import pathlib,ast,json,torch
p=pathlib.Path('work/tree-grpo/verl/utils/torch_functional.py');tree=ast.parse(p.read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_eos_mask');ns={'torch':torch};exec(compile(ast.Module(body=[f],type_ignores=[]),str(p),'exec'),ns)
rows=torch.tensor([[41,42,43,0,0,0],[51,52,2,0,0,0]])
mask=ns['get_eos_mask'](rows,eos_token=2);x={'scope':'actual f4 get_eos_mask function; toy tokens pad0 EOS2','raw_nonpad_perrow':(rows!=0).sum(1).tolist(),'response_attention_perrow':mask.sum(1).tolist(),'counterexample':'no EOS search-stop row counts 6 attention positions vs 3 nonpadding returned tokens','pad_equal_eos_limitation':'token identity cannot distinguish generated EOS from padding when pad==EOS; raw engine lengths needed for exact including EOS'}
assert x['response_attention_perrow']==[6,3] and x['raw_nonpad_perrow']==[3,3]
pathlib.Path('work/aux-gate-review/validation-cost-mask-counterexample.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x,indent=2))
