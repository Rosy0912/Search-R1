import pathlib,sys,tarfile,io,hashlib,json,subprocess
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-20261005')
EXPECTED={'tools/resource_watch.py': '0812ed859a16f9656e3c6cf8596d7895a31e678d40a7359c2d54c1efd37f268d', 'tools/native-state-fixture.py': '9ac2ef9ece3f78f0ed5a2ed71f157fec22e3e163c34388a6787b1aaa68fb2dca', 'tools/hardware.sh': '75dbac55285c4b127e1eeea35c503166346480dfc70e935d9cb1b67fc74b0e57', 'tools/native-stability-fixture.py': '1e0cb76420338c0cb21fe74e2c34779d7eeb6d2203c667d16fa18462bb173124', 'tools/ray_cli_compat.py': '2722b814ce2bc11a5f9bd6c635c89116e6700e8585faf7d50b9175aaf6a26845', 'tools/bapo-retrieval-parity.py': '0c3496c7eb2fdab56d031bee70dbbce338abebae582b96d375795680e199d370', 'tools/flash_preflight.py': 'b5be1936a34d97cc06d681ee3b3c5605b72bab5ce42dcd3f3160261086ceb299', 'tools/stage.sh': 'cd871ce82fbad6046d28733829f63f03f10fb7259908c7614ba6dca8208e8208', 'tools/stage-gpu-inner.sh': '4c546f3d8237a151e0ff0db055c4ca81cf17157bc9c08ed924b6efba5d61ee01', 'tools/native-actor-fixture.py': 'da6ea267ed1820771b5f834b4b4d2a57cbc4158f6e0916547f34243e0cc2c9e2', 'tools/verify_launch.py': 'a0678c45b2dc831043124c453de5b0c04f54d568dcf52bb9ebd668b2456d1bbf', 'tools/formal300.sbatch': '0f88dec1489ea7acb373b485554d08e3a62c9454bee7a13395390500bbf135f9', 'tools/submit.sh': 'ac7d1aba890e129d627f1db3336970c8b27e9054b4f24d43446744dd923b1dc4', 'tools/owned_processes.py': 'c7fd6fe7379b085bf20867da314eca750695dbe8b883b79b0cb25852ced55188', 'tools/hardware.sbatch': '84daee9831a3871fc9237c9de831bf07d9283c238c75c8b8b192284d85f0b800', 'tools/verify_shared_inputs.py': 'd3a7c3eda02a4ae447ec59e55fcec1d2244786532b57a3b781291e2f5e9b7484', 'tools/numerics-gpu-fixture.py': 'd9ae5b59e3ef83c600bce0050811068f05ff17ca2929168b7365e0b2ba542d6f', 'tools/monitor.sh': '3286530e84b78bf25251fc62200d0bda89c762824c1ecd1d53ede7e54e6e403c', 'tools/verify_hardware_receipt.py': '38000205d65cb891c4eb8f4b167aba0e7975d695c5867d2d7fed0056eb12fb09', 'tools/submission-worker-intent.py': '5bfed3ace4c927aa8bc70ed6d5c50a1fbc697dd171aa5c0fb3250c77c47ad514', 'tools/native-count-fixture.py': 'b40ef907be97da16b335b322983789f6522e1a2856640e8cbb976e540c1b0c6a', 'tools/cpu-tests.py': '97ef6b3083e8d2f064a59ea4bbcbc4e277aa55a8a1720cad63aeaadab83fcdcb', 'logs/tools-manifest.json': '4e97b8eda50029dad4ac3460c33cbe4aa52d0fd10e589bbe750b203b2ba6ba35', 'logs/launch-approval.json': '169649657cad4b1949dbf5c9b85d9d8b393d31dc04c56db543b09ecb18b27848', 'logs/STABILITYFIX-22-TOOLS-INDEPENDENT-REVIEW.md': '9c8f1a17baa574a920e2b8fd9b75c834f3e05586737b24cacf8f3d80c94ac8b6', 'evidence-tools/capture-hardware-stabilityfix.py': 'e913acd5804863598c49ae673380b35abed3abbfd3e30d8248fb1092422390da'}
ARCHIVE_SHA='cfac04e659e8d2d97ebc1cd0c8f3811ed9f4faeb9813429e284eaf6bf6749938'
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
