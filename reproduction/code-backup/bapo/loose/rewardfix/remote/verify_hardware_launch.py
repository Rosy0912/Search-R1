import hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-b097')
SOURCE='b09736d9d1b487a5c134e7a358761e04613cea32'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert ROOT.is_dir() and not ROOT.is_symlink()
assert subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()==SOURCE
assert not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True).strip()
record=json.loads((ROOT/'logs/hardware-launch-approval-b097.json').read_text())
assert record['status']=='PASS_INDEPENDENT_FIXED_SOURCE_LAUNCH_REVIEW'
assert record['source_commit']==SOURCE
assert record['scope']=='LAUNCH_ONLY_NOT_HARDWARE_OR_TRAINING_ACCEPTANCE'
assert len(record['evidence'])>=14
seen=set()
for item in record['evidence']:
 p=Path(item['path']).resolve()
 assert ROOT in p.parents and p.is_file(), 'Outside owned evidence'
 assert str(p) not in seen
 seen.add(str(p))
 assert digest(p)==item['sha256'],str(p)
required=['tools/stage.sh','tools/stage-gpu-inner.sh','tools/hardware.sh','tools/monitor.sh','tools/resource_watch.py','tools/owned_processes.py','tools/verify_shared_inputs.py','tools/verify_hardware_launch.py','tools/verify_hardware_receipt.py','logs/cpu-gate-b097.json','logs/cpu-full-b097.txt','logs/environment-acceptance.json','logs/shared-inputs.json','logs/isolation-setup.json','logs/common-split-manifest.json','logs/stage64-b097-proof.json','tools/flash_preflight.py','tools/bapo-retrieval-parity.py','tools/ray_cli_compat.py','tools/cpu-gate.py','tools/cpu-tests.py','tools/prepare-stage64-proof.sh','tools/parity-gpu64.sbatch','tools/submit-hardware.sh','tools/hardware-launch-hashes-b09736d.json']
assert all(str((ROOT/p).resolve()) in seen for p in required)
assert any('independent-launch-review' in Path(p).name for p in seen)
cpu=json.loads((ROOT/'logs/cpu-gate-b097.json').read_text())
assert cpu['source_commit']==SOURCE and cpu['scope']=='TARGET_LINUX_CPU_ONLY' and cpu['cuda_visible_devices']==''
assert cpu['tests_sha256']==digest(ROOT/'logs/cpu-full-b097.txt')
assert cpu['environment_sha256']==digest(ROOT/'logs/environment-acceptance.json')
txt=(ROOT/'logs/cpu-full-b097.txt').read_text()
assert '174 passed' in txt and 'skipped' not in txt
assert (ROOT/'logs/cpu-gate-source.txt').read_text().strip()==SOURCE
split=json.loads((ROOT/'logs/common-split-manifest.json').read_text())
assert split['content_identity']['sha256']=='fac19c74fe28df2016bdff12f74f035fe272bed0d65d37e9740071e674195524'
proof=json.loads((ROOT/'logs/stage64-b097-proof.json').read_text().split('BAPO_STAGE64_DERIVATION_PROOF_JSON=')[-1])
for path in [Path(proof['stage_file']),Path(proof['stage_file']).with_name('manifest.json')]:
 assert ROOT in path.resolve().parents and str(path.resolve()) in seen
assert digest(Path(proof['stage_file']))==proof['stage_sha256']=='82a5ed0d04b261031d4ee03532f0c35fc688dd2fb11c7cdaf1cc87ad0e46c34e'
assert proof['train_rows']==166222 and proof['disjoint_training_periodic_final'] is True
subprocess.run([str(ROOT/'env/bin/python'),str(ROOT/'tools/verify_shared_inputs.py')],check=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
print('FIXED_SOURCE_INDEPENDENT_LAUNCH_GATE_OK')
