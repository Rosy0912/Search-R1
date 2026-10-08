import pathlib,tarfile,hashlib,subprocess,json,os
SOURCE='733097feeae371b4efd8fd48940c46ff36da839b'
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-precision-r4-20261004')
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
EXPECTED={'tools/bapo-retrieval-parity.py': '0c3496c7eb2fdab56d031bee70dbbce338abebae582b96d375795680e199d370', 'tools/cpu-tests.py': '927d17df3d93621e94029c07727fc3acd98b5ff01236fd2318851e297f4ba4c4', 'tools/flash_preflight.py': 'b5be1936a34d97cc06d681ee3b3c5605b72bab5ce42dcd3f3160261086ceb299', 'tools/formal300.sbatch': '45a9634f51c9b9ccaec1275e73d42944559dbc489c0af75782f4d4d31bab3a67', 'tools/hardware.sbatch': 'c900761d7ad85fceff9dbba4525abdef234ab201fad9e0d910ccd338e6b7ce7d', 'tools/hardware.sh': '8dd58a5edabb66db66d8da2b066d015807a7e9bc1bfd5bcb725e037c77f0d510', 'tools/monitor.sh': '8db5ccac516d460d2d15026ee94497d8f228092be395b272d420c8e4b1d9102f', 'tools/native-actor-fixture.py': '8d342873d09892396ab1ecd8ec3f32868cc5021e5bb307ae97b0838948291998', 'tools/native-count-fixture.py': 'b40ef907be97da16b335b322983789f6522e1a2856640e8cbb976e540c1b0c6a', 'tools/native-state-fixture.py': 'b2e3e20cec1a62374d0f674ac9c1748037244818ae8ae3fe3c0b1083b9c216d8', 'tools/numerics-gpu-fixture.py': 'd0b204207648f506df746120cfc599426d4cc9f16327125df56a7452d36fb2a8', 'tools/owned_processes.py': 'c7fd6fe7379b085bf20867da314eca750695dbe8b883b79b0cb25852ced55188', 'tools/ray_cli_compat.py': '2722b814ce2bc11a5f9bd6c635c89116e6700e8585faf7d50b9175aaf6a26845', 'tools/resource_watch.py': '0e5b06c975c5346ccd10a4a8165d74f65639cb0e5a3af8e8dd41d1410b3f0ae3', 'tools/stage-gpu-inner.sh': '3fff2027baf4c12edc56afddce48fff37aa8d99d228f53a9415df191bc16b954', 'tools/stage.sh': '99fc652a7c735a643e1a43d3ee9fc3cdc680c35d5b25c30b61e0d2239473a523', 'tools/submission-worker-intent.py': '9e222657cc279e4d105e656d5eb277bba632e5e88ee86c70884a9d2a0f1f6a95', 'tools/submit.sh': 'a7ceac2352fca8a5cfa0d9d00a78fd03e24700a2a13c7aea4224871f4f481a26', 'tools/verify_hardware_receipt.py': 'f1a8515c1ec667af0530f6b365cda8499cb24036a87e0d0137bb7f27a3b11534', 'tools/verify_launch.py': '64d7d9501be0108a23f352f9d89919ec03d3cfaa3ee778865c9fcbee2ef169f2', 'tools/verify_shared_inputs.py': 'ed8391704b9b61612427997ba03ec43bec9811a7f0a327226e369cc6a00d1a1a', 'evidence-tools/capture-hardware-devicefix.py': 'c9be35a5f57da85082dabd4cc6457f458fdcd4129a10017e81119c2a75c87e91', 'evidence-tools/deploy-cpu-launcher-small.py': 'ceba9d0c0e5f607afaad76a8538bbf8d8c2e6c8b7f709597f8d1df04d91abb82', 'evidence-tools/deploy-frozen-head.py': '32906de681d284af604ab497ff992736efc3a7f7c36d4d21cb19ffbe3751da1b', 'evidence-tools/deploy-frozen-tools.py': '32906de681d284af604ab497ff992736efc3a7f7c36d4d21cb19ffbe3751da1b', 'evidence-tools/deploy-source-small.py': '8a45fa75069cd992f3dc7da14cad8f49ce503de29130625d7b5b7f989111d5ce', 'evidence-tools/native-actor-fixture.py': '8d342873d09892396ab1ecd8ec3f32868cc5021e5bb307ae97b0838948291998', 'evidence-tools/prepare-new-root-inputs.py': '76f57c0bc1a1a7f2d362a74501a42762f5cf872c21d53a4b44c5c9215c94de31', 'evidence-tools/reuse-verified-source-bundle.py': 'cdbd6aa14489175ec221290bdad6bb6cd093d22d1fba65d150caf6f27a9c33d2', 'evidence-tools/run-target-cpu-gate.py': 'db94aa16c27fb6d4aaefed998ca6b93fcdae6084d07f3bc50be2d5bf9ab40565', 'evidence-tools/step5-extract.py': '764fb22bf0ec200bbe8cb70f21bfa3cd44a6532b223eb43ad673c723de6c2d82', 'evidence-tools/test_capture_hardware_devicefix.py': 'ce91bb7385d707f1a75598291a4e9181c9db9b38f39f2e28c9dafe0e7887ba4b', 'evidence-tools/test_capture_tool_count_devicefix.py': 'cb634df021c48f36dbc14f8e9239956eae5e7384fcdea30e0cdc857606e0ad8d', 'evidence-tools/test_cpu_import_isolation.py': 'ae72ec3a670af500c729c4d31b41db8092ad690d1b61a7f1f2921bfafd09670a', 'evidence-tools/test_cpu_recorder_prefix.py': '9be959badd2835d5fd2930a2162721ec966740b87a619dacb0ea6634bf21039e', 'evidence-tools/test_cpu_source_marker_gate.py': 'fdfd4b00c4c108ffdb7bcb23f90dd3c2654e2ad19c2dd63e3097343a6dc586a4', 'evidence-tools/test_import_origin_record_gate.py': '2d916fa685b0cecb5a2ae7bc4c003b39b02054eae5eadef222ac8d7d1cf22af2', 'evidence-tools/test_native_count_gate.py': '25ec3e7b38aefd12c22e22047636bd92b041f839f4b2ca06dc69d65facff822e', 'evidence-tools/test_precision_oracle.py': '0776004c8db0e3934b68a3150d6a2ce00ca722de06f7dfd2d129164e3b92f270', 'evidence-tools/test_precision_probe_gate.py': 'ceb7e1f4fa6a2d7462c166191278670a4aae711db4f63657fcb0bb2293386b26', 'evidence-tools/test_state_snapshot_gate.py': '5876981b5b6c9595a3b6a69e92ecc4a7594f12a1bed38eab837560a7604c3685', 'evidence-tools/test_step5_extract.py': '3e72b646458b5ea725e1655975d2b8b9883da7d093778b06522d2bfac111fd56', 'evidence-tools/test_step5_independent_review.py': '090891a462e5421bcc4c4330d03771d0191e613d934daf8e812eda2ae905ac78', 'evidence-tools/test_step5_tied_math.py': '3e82e8e3318c8f80cca424b791ff8ad54559a1cd2b5962a04fd2f1ece91f7a1d', 'evidence-tools/test_tensordict_device_api_capture.py': '649f5ed427f8f7344ba1f1b91eaf273d03aee41803d58f650198d28ef60e7f5d', 'evidence-tools/watch-terminal.py': '15cfe2686cf9eea8d464c962e726dd6108a3a91cae70b7102adf50577b373e92', 'logs/tools-manifest.json': '622dfcf5f529a8127f7ed55b28fd9e6c9f5f12cd2d6f41a0766d4639f623a520', 'logs/launch-approval.json': 'f5f5212d82aec616d5702568a31134545adfdbd5ba087b69fcc0dad18877389e', 'logs/INDEPENDENT-PRECISION-R4-LAUNCH-TOOLS-REVIEW.md': 'a21eda0f306822ca9aa65c16de09fbf32de51dea2d965a1357af8278e1e8cdef'}
ARCHIVE_SHA='76ce805c32c63acbd0524ac7719100057cedb1e51825ef811b80dab9d02e2405'
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
