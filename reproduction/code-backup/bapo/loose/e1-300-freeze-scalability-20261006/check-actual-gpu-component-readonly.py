"""Execute frozen raw validator only; no CUDA, receipt, signals or target writes."""
import ast,pathlib,json,hashlib,math,itertools,collections,csv
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-freeze-scalability-20261006')
run=root/'runs/bapo-instruct300-hardware-3263';tool=root/'tools/verify_hardware_receipt.py'
raw=tool.read_bytes();expected='2016ff3dcd38c31ea69226d29754123e8af686a02cbb2969f287a2466cbf8c5d'
assert hashlib.sha256(raw).hexdigest()==expected
parsed=ast.parse(raw);names={'sha','finite','validate_freeze_scalability'}
selected=[n for n in parsed.body if isinstance(n,ast.FunctionDef)and n.name in names]
assert {n.name for n in selected}==names
namespace={k:v for k,v in globals().items()};exec(compile(ast.Module(body=selected,type_ignores=[]),str(tool),'exec'),namespace)
paths=list((run/'freeze-scalability-gpu').iterdir())+[root/'tools/freeze_scalability_gpu_fixture.py',root/'tools/resource_watch.py']
assert all(p.is_file()and not p.is_symlink()for p in paths)
bound={str(p):namespace['sha'](p)for p in paths}
rows=list(csv.DictReader((run/'gpus-before.csv').read_text().splitlines(),skipinitialspace=True))
assert len(rows)==8 and {int(x['index'])for x in rows}==set(range(8))
uuids={x['uuid'].strip()for x in rows};assert len(uuids)==8
result=namespace['validate_freeze_scalability'](run,root,'b7119791b86951727cea36a1881e8a525c932494',3263,bound,uuids)
assert hashlib.sha256(tool.read_bytes()).hexdigest()==expected
assert all(namespace['sha'](pathlib.Path(p))==h for p,h in bound.items())
print(json.dumps({'status':'ACTUAL_TARGET_RAW_GPU_COMPONENT_VALIDATOR_PASS_NOT_FULL_HARDWARE_ACCEPTANCE','job_id':3263,'validator_path':str(tool),'validator_sha256':expected,'function_origin':namespace['validate_freeze_scalability'].__code__.co_filename,'proof':result,'bound_files':len(bound),'actual_gpu_initialized_here':False,'hardware_log_may_grow':True,'full_hardware_receipt_signed':False},sort_keys=True))
