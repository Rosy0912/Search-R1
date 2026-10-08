import hashlib,json,time
from pathlib import Path
p=Path('tests/baselines/test_instruct300_actor.py');source=p.read_text()
# pytest is an unused import in this file, absent from the existing remote env.
# Invoke its two plain assertion functions unchanged, without installing packages.
assert source.count('\nimport pytest\n')==1
scope={'__name__':'cpu_direct_actor_regression'}
exec(compile(source.replace('\nimport pytest\n','\n'),str(p),'exec'),scope)
start=time.time()
for name in ('test_actual_actor_fixed128_micro2_16_128_same_target_and24_steps','test_actual_recovery_rank480_mini48_micro2_performs10_updates'):
 scope[name]()
 print('PASS',name,flush=True)
print(json.dumps(dict(status='ACTUAL_ENV_CPU_ACTOR_REGRESSION_PASS',tests=2,wall_seconds=time.time()-start,test_source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),runner='Direct assertion functions; unused pytest import omitted only in test driver; production code unchanged; CUDA_VISIBLE_DEVICES empty')))
