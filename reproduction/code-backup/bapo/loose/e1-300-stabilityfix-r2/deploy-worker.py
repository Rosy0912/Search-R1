import pathlib,sys,tarfile,io,hashlib,json,subprocess
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005')
EXPECTED={'tools/resource_watch.py': '0812ed859a16f9656e3c6cf8596d7895a31e678d40a7359c2d54c1efd37f268d', 'tools/native-state-fixture.py': '9ac2ef9ece3f78f0ed5a2ed71f157fec22e3e163c34388a6787b1aaa68fb2dca', 'tools/hardware.sh': '23579f02dd0b4c55a778a6a9d77b899d3c52e1406070f4803fc085301dd1d58c', 'tools/native-stability-fixture.py': 'a92dbf8727f3a323591b9330f66bcf8e4ef5e3b2c581a742fce47746de343e63', 'tools/ray_cli_compat.py': '2722b814ce2bc11a5f9bd6c635c89116e6700e8585faf7d50b9175aaf6a26845', 'tools/bapo-retrieval-parity.py': '0c3496c7eb2fdab56d031bee70dbbce338abebae582b96d375795680e199d370', 'tools/flash_preflight.py': 'b5be1936a34d97cc06d681ee3b3c5605b72bab5ce42dcd3f3160261086ceb299', 'tools/stage.sh': '60dd52ff738fb4894595146f5d9c00375e83c7691e9a93c79f75083bd2e99d23', 'tools/stage-gpu-inner.sh': 'b6ce08db15969b323eff1cdb0ef7565f824936f7e0d324fbbc34d16cfb8f357b', 'tools/native-actor-fixture.py': 'eef9e80542bbc0fb72dbd4e6f94246b6fb34c2aac13dc079c24f27e2ee781309', 'tools/verify_launch.py': '2bb2ab09b5546a5f805f35f0996fdaa53831ba31a136ab58d9322618d3db42d0', 'tools/formal300.sbatch': '1f9014c24748a7cb44779284fe3b499e110f1188cb396fe55edec9e1118a5dd8', 'tools/submit.sh': '34fdb8380fba79ca426e7080261e222a6083701d2ae9d0c06cdfc99e766c1e30', 'tools/owned_processes.py': 'c7fd6fe7379b085bf20867da314eca750695dbe8b883b79b0cb25852ced55188', 'tools/hardware.sbatch': '6900d9c63ef631c29b148b68acc1e0fbbbed9d2b938dfc0a490a66afc3976e1f', 'tools/verify_shared_inputs.py': '87e15cbb2ca78f1542e049d1b0f1ea06c9e1f0a879b530f8dd507c8b47463773', 'tools/numerics-gpu-fixture.py': '6534e667562cc6dfe399bc144e730d5a821ac1321a074d0aa66eddfad68c8e61', 'tools/monitor.sh': 'be348a393fcda911fb6f853acd50cda0b5173e6f56f5f32e6bb4825e25a291e4', 'tools/verify_hardware_receipt.py': 'a3b5471fd911f0b7fe2802d21707821a63652fa96c638615e11577032feabee4', 'tools/submission-worker-intent.py': '6901289349733c8e482d0fbd2e966a059a633725f69951e9221bdcb9a18e90a1', 'tools/native-count-fixture.py': 'b40ef907be97da16b335b322983789f6522e1a2856640e8cbb976e540c1b0c6a', 'tools/cpu-tests.py': '97ef6b3083e8d2f064a59ea4bbcbc4e277aa55a8a1720cad63aeaadab83fcdcb', 'logs/tools-manifest.json': 'f74329d955df6dbf02927424e8e5c225e4c958c776c50f5aaea37f2cac3cd9c6', 'logs/launch-approval.json': '8cb6d8a87705d5c34f8aed5118feaf7d96b9cd085a73847c099db33e8930f9d2', 'logs/STABILITYFIX-R2-LAUNCH-INDEPENDENT-REVIEW.md': '1df16b30fa84e777262be5db2da217d8ea1c3952efee39ca5527e84fc8b09ca9', 'evidence-tools/capture-hardware-stabilityfix.py': '4fd38c7233f84311ce6f5360a4a097a7460a5fc60e230939e851d9ea1824804f'}
ARCHIVE_SHA='172d0820a2a900ba4b682571c13afdc9800c824cd6db3c2812ffeb76a2796184'
KIND='worker'
raw=sys.stdin.buffer.read();assert hashlib.sha256(raw).hexdigest()==ARCHIVE_SHA
if KIND=='head':
 if not ROOT.exists():ROOT.mkdir(parents=True)
assert ROOT.is_dir() and not ROOT.is_symlink() and ROOT.resolve()==ROOT
if KIND=='worker':
 assert subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()=='39cdcdd1404213b897bb0de99cc52d72707682f0'
 assert not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True)
 assert not list((ROOT/'runs').iterdir())
 record=json.loads((ROOT/'logs/cpu-gate.json').read_text());assert record['passed']==305 and record['exit_code']==record['skipped']==record['failed']==0
 destination=ROOT
else:
 destination=ROOT/'scheduler-head';assert not destination.exists()
payload={}
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz')as bundle:
 members=bundle.getmembers();assert len(members)==len(EXPECTED) and {m.name for m in members}==set(EXPECTED)
 for m in members:
  q=pathlib.PurePosixPath(m.name);assert m.isfile() and not q.is_absolute() and '..'not in q.parts
  data=bundle.extractfile(m).read();assert hashlib.sha256(data).hexdigest()==EXPECTED[m.name]
  payload[m.name]=(data,m.mode)
for name,(data,mode)in payload.items():
 p=destination/name;assert destination in p.parents
 if p.exists()or p.is_symlink():assert p.is_file()and not p.is_symlink()and hashlib.sha256(p.read_bytes()).hexdigest()==EXPECTED[name]
 if KIND=='worker':assert p.parent.resolve()==p.parent
if KIND=='head':destination.mkdir()
for name,(data,mode)in payload.items():
 p=destination/name;p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():p.open('xb').write(data)
 p.chmod(mode);assert hashlib.sha256(p.read_bytes()).hexdigest()==EXPECTED[name]
print(json.dumps({'status':'VERIFIED_DEPLOYMENT_NO_SUBMISSION','kind':KIND,'files':len(payload),'root':str(ROOT)}))
