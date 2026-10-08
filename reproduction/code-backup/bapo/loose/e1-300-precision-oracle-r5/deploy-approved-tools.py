import pathlib,tarfile,hashlib,subprocess,json,os
SOURCE='733097feeae371b4efd8fd48940c46ff36da839b'
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-precision-r5-20261004')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def decode_bundle(path,expected):
 result={}
 with tarfile.open(path,'r:gz')as t:
  members=t.getmembers();assert len(members)==len(expected)and{m.name for m in members}==set(expected)
  for m in members:
   q=pathlib.PurePosixPath(m.name);assert m.isfile()and not q.is_absolute()and '..'not in q.parts and all(x not in ('','.','..')for x in q.parts)
   b=t.extractfile(m).read();assert hashlib.sha256(b).hexdigest()==expected[m.name]
   result[m.name]=(b,m.mode&0o777)
 return result
EXPECTED={'tools/bapo-retrieval-parity.py': '0c3496c7eb2fdab56d031bee70dbbce338abebae582b96d375795680e199d370', 'tools/cpu-tests.py': '927d17df3d93621e94029c07727fc3acd98b5ff01236fd2318851e297f4ba4c4', 'tools/flash_preflight.py': 'b5be1936a34d97cc06d681ee3b3c5605b72bab5ce42dcd3f3160261086ceb299', 'tools/formal300.sbatch': 'f49136220f5ea206a855a92ff118aecfddbb2f2909fda9bfd4ce90a49aa960f7', 'tools/hardware.sbatch': '069631588f4665ec0c873ed8f6b19333f2efe858794a024fc076f99dbd1277ee', 'tools/hardware.sh': '053e7a81103462d584bd613d33f67d74a4ca039b0f5eea925fc99fe314d6436d', 'tools/monitor.sh': 'a3888a0916a83d9e04c4f8ccf14d1b6e5bca2c02cbb24558fc0c8d144f0bb3f2', 'tools/native-actor-fixture.py': '45ea575e2e1b51c59d534e22daa527b129af99e64a17e8b8dfaaee6048290630', 'tools/native-count-fixture.py': 'b40ef907be97da16b335b322983789f6522e1a2856640e8cbb976e540c1b0c6a', 'tools/native-state-fixture.py': 'b2e3e20cec1a62374d0f674ac9c1748037244818ae8ae3fe3c0b1083b9c216d8', 'tools/numerics-gpu-fixture.py': '05c82debec2551c12e1e89cdcd07c77b5d955ac71d14953cc3b1900fe3d52e55', 'tools/owned_processes.py': 'c7fd6fe7379b085bf20867da314eca750695dbe8b883b79b0cb25852ced55188', 'tools/ray_cli_compat.py': '2722b814ce2bc11a5f9bd6c635c89116e6700e8585faf7d50b9175aaf6a26845', 'tools/resource_watch.py': '0e5b06c975c5346ccd10a4a8165d74f65639cb0e5a3af8e8dd41d1410b3f0ae3', 'tools/stage-gpu-inner.sh': '1f1aa89bfe31d6e97ab46d1204870c751aa27097495fe5a1d7f32d071602a59a', 'tools/stage.sh': '7c0c7179d2c0025557db8dfd4140961f9bafc7c25a24b368f99d9f441250ea78', 'tools/submission-worker-intent.py': 'b5849a248dd8d481375c15d28ca9590d47826b43e265e4e30106466dd965a964', 'tools/submit.sh': '83f5083f1e2d5d4b6630d6c7bd7762f14898f6348ed433759d42f2c24c029907', 'tools/verify_hardware_receipt.py': 'c5cb72ef60be6bfa0633ef8f7b9e68abeac14ece12516f24753aac4505f41668', 'tools/verify_launch.py': '54f3b9f64f6707783a12194305e10abc7a3b793b50a3beaddaa756fc50b9c9c3', 'tools/verify_shared_inputs.py': 'f69e528ac6bf21677fb04e363c975a1dc9f14218b20fe4769a484b3a0496aada', 'evidence-tools/capture-hardware-devicefix.py': 'ba33019cd4ded6286be19677faf8b486af027737666dea25043545cb8d119d27', 'evidence-tools/deploy-cpu-launcher-small.py': 'e8eccc2fadbb84f1a2bfd67630b1c610ddb4a19a366d646a98a9060fbd5c4dbe', 'evidence-tools/deploy-frozen-head.py': '4bf330bdebe4a4ff37794e93fb6e630099acdc50387089773a65032c2cf324f1', 'evidence-tools/deploy-frozen-tools.py': '4bf330bdebe4a4ff37794e93fb6e630099acdc50387089773a65032c2cf324f1', 'evidence-tools/deploy-source-small.py': '54c7121b93a05a1b37b00fb1333278e7030a689d48d361e93d16a9dfb0061f84', 'evidence-tools/native-actor-fixture.py': '45ea575e2e1b51c59d534e22daa527b129af99e64a17e8b8dfaaee6048290630', 'evidence-tools/prepare-new-root-inputs.py': 'e22bfd28733c7c1b5a85209aa1985ceed7b4c78dea2f92dbb45d646476331d18', 'evidence-tools/reuse-verified-source-bundle.py': '63a3b157e923edaf9bed2219637b8d0035186f41d891353ef725c477e22dfc6d', 'evidence-tools/run-target-cpu-gate.py': '0c082b945edc2c7b978c267ce8cfbb8f32c110e9016db3cf7b6babbeb01b5bca', 'evidence-tools/step5-extract.py': '4b494edacf3c50e5083395cc38355f03b63757a08e662f46de0fa225b28cb27f', 'evidence-tools/test_capture_hardware_devicefix.py': 'ce91bb7385d707f1a75598291a4e9181c9db9b38f39f2e28c9dafe0e7887ba4b', 'evidence-tools/test_capture_tool_count_devicefix.py': '93764e08187e7c5a1608cdeebb57cbf82c9790ee94b2b1ec2a083eff0e78cf6a', 'evidence-tools/test_cpu_import_isolation.py': 'ae72ec3a670af500c729c4d31b41db8092ad690d1b61a7f1f2921bfafd09670a', 'evidence-tools/test_cpu_recorder_prefix.py': 'fbc2987c4bdaab70b364c4c8670bf823bfea3371bf0a0d202910af75792d924e', 'evidence-tools/test_cpu_source_marker_gate.py': 'fdfd4b00c4c108ffdb7bcb23f90dd3c2654e2ad19c2dd63e3097343a6dc586a4', 'evidence-tools/test_import_origin_record_gate.py': '2d916fa685b0cecb5a2ae7bc4c003b39b02054eae5eadef222ac8d7d1cf22af2', 'evidence-tools/test_native_count_gate.py': '25ec3e7b38aefd12c22e22047636bd92b041f839f4b2ca06dc69d65facff822e', 'evidence-tools/test_precision_oracle.py': '73c2aeb3618a6119fe1cc7572ad4759ef909c721d1caa73eac89ae4594a6a8f3', 'evidence-tools/test_precision_probe_gate.py': 'c1211ee82f558ea24ba466b68ac427d67cb6cdcecdb1dfdf57830c1cff3e7a09', 'evidence-tools/test_state_snapshot_gate.py': '5876981b5b6c9595a3b6a69e92ecc4a7594f12a1bed38eab837560a7604c3685', 'evidence-tools/test_step5_extract.py': '3e72b646458b5ea725e1655975d2b8b9883da7d093778b06522d2bfac111fd56', 'evidence-tools/test_step5_independent_review.py': '090891a462e5421bcc4c4330d03771d0191e613d934daf8e812eda2ae905ac78', 'evidence-tools/test_step5_tied_math.py': '3e82e8e3318c8f80cca424b791ff8ad54559a1cd2b5962a04fd2f1ece91f7a1d', 'evidence-tools/test_tensordict_device_api_capture.py': '649f5ed427f8f7344ba1f1b91eaf273d03aee41803d58f650198d28ef60e7f5d', 'evidence-tools/watch-terminal.py': 'a72aced457880e8956a346a9f4d55e3f4ed0113ca20f25af278e319e102ae2ff', 'logs/tools-manifest.json': '562b936575e077211308900c08ebaff4bcee3884d99e9cdc4bbce9a246cf83e5', 'logs/launch-approval.json': '45647da2aa51cc67d9e76e39d562c43d9a25e55ee30edb37bc1dcc81f447dd3c', 'logs/INDEPENDENT-PRECISION-R5-LAUNCH-TOOLS-REVIEW.md': '63751ce74ca552a67c2e0f1d2d8bfa54731de653f90c374fd4cad07e3451906b'}
ARCHIVE_SHA='2f0d6b4dc7ead79807f30a83a6503a70ad6ace7e31d692d6642bb7e02e37762b'
p=ROOT/'approved-launch-deploy.tar.gz'
assert ROOT.is_dir()and not ROOT.is_symlink()and ROOT.resolve()==ROOT
assert sha(p)==ARCHIVE_SHA and not p.is_symlink()
assert (ROOT/'runs').is_dir()and not(ROOT/'runs').is_symlink()and not list((ROOT/'runs').iterdir())
assert not any((ROOT/'logs').glob('submission-intent-*.json'))
assert subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()==SOURCE
assert not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True).strip()
payload=decode_bundle(p,EXPECTED)
assert len(payload)==49 and len([x for x in payload if x.startswith('tools/')])==21
for directory in ('tools','logs','evidence-tools'):
 q=ROOT/directory;assert q.is_dir()and not q.is_symlink()and q.resolve()==q
assert {x.name for x in (ROOT/'tools').iterdir()}<={pathlib.PurePosixPath(x).name for x in payload if x.startswith('tools/')}
# Validate ALL destinations before writing the first approved byte.
for name,(b,mode)in payload.items():
 dst=ROOT/name;assert ROOT in dst.resolve().parents and dst.parent.resolve()==dst.parent
 if dst.exists()or dst.is_symlink():assert dst.is_file()and not dst.is_symlink()and sha(dst)==EXPECTED[name]
approval=json.loads(payload['logs/launch-approval.json'][0]);manifest=json.loads(payload['logs/tools-manifest.json'][0])
assert approval['root']==str(ROOT)and approval['source_commit']==SOURCE and manifest['source_commit']==SOURCE
assert sha(ROOT/'logs/cpu-gate.json')==approval['cpu_gate_record_sha256']
record=json.loads((ROOT/'logs/cpu-gate.json').read_text());assert record['sys_prefix']==str(ROOT/'env')and record['passed']==270 and record['skipped']==0 and record['failed']==0 and record['exit_code']==0
assert sha(ROOT/'logs/cpu-gate.log')==record['log_sha256']and sha(ROOT/'logs/cpu-gate-source.txt')==record['source_marker_sha256']
for name,(b,mode)in payload.items():
 dst=ROOT/name
 if not dst.exists():
  with dst.open('xb')as f:f.write(b)
  dst.chmod(mode)
 assert sha(dst)==EXPECTED[name]
assert {x.name for x in (ROOT/'tools').iterdir()}==set(manifest['files'])
print(json.dumps({'status':'APPROVED_NEWROOT_TOOLS_DEPLOYED_NO_SUBMISSION_NO_GPU_ACCEPTANCE','source_commit':SOURCE,'files':49,'archive_sha256':ARCHIVE_SHA}),flush=True)
