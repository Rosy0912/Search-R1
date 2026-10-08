from pathlib import Path
old='/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-b097';new='/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-numeric'
source='1cceda2f7e201de3f68c876bb9a3b895563c20ce'
r=Path('work/numericfix/remote')
for p in Path('work/rewardfix/remote').iterdir():
 if not p.is_file() or p.name=='hardware-launch-hashes-b09736d.json':continue
 name=p.name
 text=p.read_text().replace(old,new).replace('b09736d9d1b487a5c134e7a358761e04613cea32',source)
 text=text.replace('b09736d','numeric-v1').replace('b097','numeric').replace('B097','NUMERIC')
 if name=='setup-isolated.py':text=text.replace("source_bundle_sha256", "source_bundle_sha256")
 if name=='cpu-gate.py':text=text.replace('174 passed','183 passed')
 if name=='verify_hardware_launch.py':
  text=text.replace('174 passed','183 passed')
  text=text.replace("'tools/hardware-launch-hashes-numeric-v1.json']", "'tools/hardware-launch-hashes-numeric-v1.json','tools/numerics-gpu-fixture.py']")
 if name=='hardware.sh':
  line='OMP_NUM_THREADS=1 python -m torch.distributed.run --nnodes=1 --master_addr=127.0.0.1 --master_port="$port" --nproc_per_node=8 "$ROOT/repo/reproduction/h20/cuda_preflight.py"'
  assert line in text
  text=text.replace(line,line+'\nexport NUMERIC_FIXTURE_OUTPUT="$RUN/numeric-fixture"\nOMP_NUM_THREADS=1 python -m torch.distributed.run --nnodes=1 --master_addr=127.0.0.1 --master_port="$port" --nproc_per_node=8 "$ROOT/tools/numerics-gpu-fixture.py"')
 if name=='verify_hardware_receipt.py':
  text=text.replace("'cpu-gate.py', 'prepare-stage64-proof.sh'}", "'cpu-gate.py', 'prepare-stage64-proof.sh','numerics-gpu-fixture.py'}")
  needle="    if (run / 'source-commit.txt').read_text().strip() != SOURCE:"
  extra="""    for rank in range(8):
        path=run/'numeric-fixture'/('rank-'+str(rank)+'.json')
        if str(path) not in seen:raise ValueError('Real numeric rank not directly bound')
        numeric=json.loads(path.read_text())
        if numeric.get('status')!='PASS' or numeric.get('source_commit')!=SOURCE or numeric.get('rank')!=rank or numeric.get('world')!=8:raise ValueError('Real numeric rank identity')
        if len(numeric.get('records',[]))!=22:raise ValueError('Real numeric rank cases incomplete')
        if Path(numeric.get('gradient_norm_file','')).resolve()!=root/'repo/search_r1/baselines/gradient_norm.py' or Path(numeric.get('core_algos_file','')).resolve()!=root/'repo/verl/trainer/ppo/core_algos.py':raise ValueError('Numeric function imported wrong source')
"""
  assert needle in text;text=text.replace(needle,extra+needle)
 (r/name).write_text(text)
p=r/'numerics-gpu-fixture.py';text=p.read_text().replace('FIXED_SOURCE_PENDING',source);p.write_text(text)
