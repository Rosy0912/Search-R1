import pathlib,tarfile,hashlib,subprocess,json,os
SOURCE='733097feeae371b4efd8fd48940c46ff36da839b'
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-monitor-r6-20261004')
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

def verify_monitor_cpu(approval):
 bound=approval['target_monitor_cpu'];d=ROOT/'temporary-evidence/monitor22-actual-linux-20261004'
 assert d.is_dir()and not d.is_symlink()and d.resolve()==d
 p=d/'monitor-cpu-gate.json';log=d/'monitor-cpu-gate.log'
 assert p.is_file()and not p.is_symlink()and sha(p)==bound['record_sha256']
 assert log.is_file()and not log.is_symlink()and sha(log)==bound['log_sha256']
 r=json.loads(p.read_text())
 assert r['root']==str(ROOT)and r['source_commit']==SOURCE and r['sys_prefix']==str(ROOT/'env')and r['executable']==str(ROOT/'env/bin/python')
 assert r['status']=='PASS_ACTUAL_TARGET_LINUX_MONITOR_CPU'and r['scope']=='MONITOR_CPU_REGRESSION_ONLY_NOT_GPU_OR_TRAINING_ACCEPTANCE'
 assert r['hostname'].split('.')[0]=='h100-5'and r['actual_tests']==22 and r['expected_tests']==22 and r['exit_code']==0 and r['cuda_visible_devices']==''
 assert r['log_path']==str(log)and r['evidence_directory']==str(d)and r['log_sha256']==bound['log_sha256']
 assert bound['sys_prefix']==r['sys_prefix']and bound['actual_tests']==22 and bound['exit_code']==0 and r['files_sha256']==bound['files_sha256']
 assert set(r['files_sha256'])=={'tools/resource_watch.py','test_resource_watch_retry.py','hardware3171-lifecycle-samples.json'}
 for name,value in r['files_sha256'].items():
  path=d/name;assert path.is_file()and not path.is_symlink()and path.parent.resolve()==path.parent and sha(path)==value
  signed=approval['tools_sha256']['resource_watch.py']if name.startswith('tools/')else approval['extra_helpers_sha256'][name]
  assert signed==value

EXPECTED={'evidence-tools/capture-hardware-devicefix.py': '0507e3206aa4af0d425c1d00af94d44193ffaff601573176e36308616c9a48d1', 'evidence-tools/deploy-cpu-launcher-small.py': 'fa103346c7721dbedd235eb44b084ad09c340b24b23e1738385bcd75b0961055', 'evidence-tools/deploy-frozen-head.py': '4bf330bdebe4a4ff37794e93fb6e630099acdc50387089773a65032c2cf324f1', 'evidence-tools/deploy-frozen-tools.py': '4bf330bdebe4a4ff37794e93fb6e630099acdc50387089773a65032c2cf324f1', 'evidence-tools/deploy-source-small.py': '2e4a1d556294cbb863acb2d899683d70a8df5a96d31e750a78412be157b25a0e', 'evidence-tools/hardware3171-lifecycle-samples.json': 'b005f4fb18da8a7c7ba44f2546cc4f5fd05d72e327fcf9b9d6708893fa11ddef', 'evidence-tools/native-actor-fixture.py': 'cd03d6782f35a03c7e9279bb3d49630387bbcd2dc5e7bb4b7cdae1cb48514fe1', 'evidence-tools/prepare-new-root-inputs.py': 'fd11481e206e15c27b2f89e3344518f18e6df1a5c0fc59decde26c2e8616c0fb', 'evidence-tools/reuse-verified-source-bundle.py': '8defe25f27f14261bac2fbbff76401aa226c3ef11ca1f30b5fcf34838bad650a', 'evidence-tools/run-target-cpu-gate.py': '05063b49e3c31ec2836a9b8492e7e50a8b856ec3901e2076c6b57464dff0b291', 'evidence-tools/step5-extract.py': '563adfa7f0d3ffecc0a5cbf1b9c95ecdc90ee23c5e8d7c8f9a68792bf607f9e6', 'evidence-tools/test_capture_hardware_devicefix.py': 'ce91bb7385d707f1a75598291a4e9181c9db9b38f39f2e28c9dafe0e7887ba4b', 'evidence-tools/test_capture_tool_count_devicefix.py': '51c15f793ec287c32ff8f576a689e39c50866c27f67b08742c55f16a4db60030', 'evidence-tools/test_cpu_import_isolation.py': 'ae72ec3a670af500c729c4d31b41db8092ad690d1b61a7f1f2921bfafd09670a', 'evidence-tools/test_cpu_recorder_prefix.py': 'f668d1a46733e78f41ace37666eec20072ffec9dfadb1bf7bee282e9ae7dfc1d', 'evidence-tools/test_cpu_source_marker_gate.py': 'fdfd4b00c4c108ffdb7bcb23f90dd3c2654e2ad19c2dd63e3097343a6dc586a4', 'evidence-tools/test_import_origin_record_gate.py': '2d916fa685b0cecb5a2ae7bc4c003b39b02054eae5eadef222ac8d7d1cf22af2', 'evidence-tools/test_native_count_gate.py': '25ec3e7b38aefd12c22e22047636bd92b041f839f4b2ca06dc69d65facff822e', 'evidence-tools/test_precision_oracle.py': '73c2aeb3618a6119fe1cc7572ad4759ef909c721d1caa73eac89ae4594a6a8f3', 'evidence-tools/test_precision_probe_gate.py': 'c1211ee82f558ea24ba466b68ac427d67cb6cdcecdb1dfdf57830c1cff3e7a09', 'evidence-tools/test_resource_watch_retry.py': 'cba621d717ddbcdc60edf7bea6ec2c743b55f2e3ad82a56186be3bba0b61894c', 'evidence-tools/test_state_snapshot_gate.py': '5876981b5b6c9595a3b6a69e92ecc4a7594f12a1bed38eab837560a7604c3685', 'evidence-tools/test_step5_extract.py': '3e72b646458b5ea725e1655975d2b8b9883da7d093778b06522d2bfac111fd56', 'evidence-tools/test_step5_independent_review.py': '090891a462e5421bcc4c4330d03771d0191e613d934daf8e812eda2ae905ac78', 'evidence-tools/test_step5_tied_math.py': '3e82e8e3318c8f80cca424b791ff8ad54559a1cd2b5962a04fd2f1ece91f7a1d', 'evidence-tools/test_tensordict_device_api_capture.py': '649f5ed427f8f7344ba1f1b91eaf273d03aee41803d58f650198d28ef60e7f5d', 'evidence-tools/watch-terminal.py': 'e63e8cb7c426c64facc7ba24d8d0d34600eece51287b8e6dd9ee741081c1c5ac', 'logs/INDEPENDENT-MONITOR-R6-LAUNCH-TOOLS-REVIEW.md': '1e13c3350d82ccbb96ebd555034d98e112ffdfd36a9f5bbe6513a707a25551d6', 'logs/launch-approval.json': '3bad3597c818da5d9c59d1c2eab329a067e5e21bbfe10e0f26403695f4fc72bb', 'logs/tools-manifest.json': '5f69991bec83ed84f4ceade9c1429d35049bf3e0f0134363abf0e573cdf59657', 'tools/bapo-retrieval-parity.py': '0c3496c7eb2fdab56d031bee70dbbce338abebae582b96d375795680e199d370', 'tools/cpu-tests.py': '927d17df3d93621e94029c07727fc3acd98b5ff01236fd2318851e297f4ba4c4', 'tools/flash_preflight.py': 'b5be1936a34d97cc06d681ee3b3c5605b72bab5ce42dcd3f3160261086ceb299', 'tools/formal300.sbatch': 'fd7b8f89f3fedd9ddb9e60ecd4339f1f5367d136a7df5a6ba1c86c94264aff30', 'tools/hardware.sbatch': '9ff9eabcfa9eb7fbb977dc1d4b03adf0c12775550d083dd182c68ddc0f35def4', 'tools/hardware.sh': 'a76ed1c0dc60f06bd4f526b61ba22fa9b28ecaf4da2a3de0f5f052b9de806bcb', 'tools/monitor.sh': '58c683e495ca5dae7293b7455ec07c7e2f0d10c7239fe735519efac8fb41d7df', 'tools/native-actor-fixture.py': 'cd03d6782f35a03c7e9279bb3d49630387bbcd2dc5e7bb4b7cdae1cb48514fe1', 'tools/native-count-fixture.py': 'b40ef907be97da16b335b322983789f6522e1a2856640e8cbb976e540c1b0c6a', 'tools/native-state-fixture.py': 'b2e3e20cec1a62374d0f674ac9c1748037244818ae8ae3fe3c0b1083b9c216d8', 'tools/numerics-gpu-fixture.py': '295916ae6a46dea30b73d45901215c797f5d2f7faa277045308c4a34da1f1ff3', 'tools/owned_processes.py': 'c7fd6fe7379b085bf20867da314eca750695dbe8b883b79b0cb25852ced55188', 'tools/ray_cli_compat.py': '2722b814ce2bc11a5f9bd6c635c89116e6700e8585faf7d50b9175aaf6a26845', 'tools/resource_watch.py': '0812ed859a16f9656e3c6cf8596d7895a31e678d40a7359c2d54c1efd37f268d', 'tools/stage-gpu-inner.sh': '3f656d5beee036042297cd8826a759fcff9d84527370adfed9292cf692a94321', 'tools/stage.sh': '1c6da8d6de20f619df069a8051c78e949b28b28e682a323f8802599641c91455', 'tools/submission-worker-intent.py': 'aeaba73400b14e01b0ddd17ed488ac2e5d5fc9c949cebad987cfcf472212dd71', 'tools/submit.sh': '92b422c74a98c28bd78de86d97b877c9dfa93855dd01f35f859f839ca2a28edc', 'tools/verify_hardware_receipt.py': 'fc300987c305e300f3508a8b556336e569055d61f4ab9e2191e010876f099387', 'tools/verify_launch.py': 'b65acf97095d31392b07e30bbf747d5b6cf469382484e3d8016bd4d438f3a9f7', 'tools/verify_shared_inputs.py': 'a9ca1bf955547af39b805d4a38be4b8aff06d5dd0cec0c2b56628f4113434d9d'}
ARCHIVE_SHA='b6045a34057ff545995c8748a5b326907776d6e1eb983b189e5f315863397a65'
p=ROOT/'approved-launch-deploy.tar.gz'
assert ROOT.is_dir()and not ROOT.is_symlink()and ROOT.resolve()==ROOT
assert sha(p)==ARCHIVE_SHA and not p.is_symlink()
assert (ROOT/'runs').is_dir()and not(ROOT/'runs').is_symlink()and not list((ROOT/'runs').iterdir())
assert not any((ROOT/'logs').glob('submission-intent-*.json'))
assert subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()==SOURCE
assert not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True).strip()
payload=decode_bundle(p,EXPECTED)
assert len(payload)==51 and len([x for x in payload if x.startswith('tools/')])==21
for directory in ('tools','logs','evidence-tools'):
 q=ROOT/directory;assert q.is_dir()and not q.is_symlink()and q.resolve()==q
assert {x.name for x in (ROOT/'tools').iterdir()}<={pathlib.PurePosixPath(x).name for x in payload if x.startswith('tools/')}
# Validate ALL destinations before writing the first approved byte.
for name,(b,mode)in payload.items():
 dst=ROOT/name;assert ROOT in dst.resolve().parents and dst.parent.resolve()==dst.parent
 if dst.exists()or dst.is_symlink():assert dst.is_file()and not dst.is_symlink()and sha(dst)==EXPECTED[name]
approval=json.loads(payload['logs/launch-approval.json'][0]);manifest=json.loads(payload['logs/tools-manifest.json'][0])
assert approval['root']==str(ROOT)and approval['source_commit']==SOURCE and manifest['source_commit']==SOURCE and manifest['root']==str(ROOT)
assert approval['tools_sha256']==manifest['files']
assert {'tools/'+n for n in manifest['files']}=={n for n in EXPECTED if n.startswith('tools/')}
assert {'evidence-tools/'+n for n in approval['extra_helpers_sha256']}=={n for n in EXPECTED if n.startswith('evidence-tools/')}
assert sha(ROOT/'logs/cpu-gate.json')==approval['cpu_gate_record_sha256']
record=json.loads((ROOT/'logs/cpu-gate.json').read_text());assert record['sys_prefix']==str(ROOT/'env')and record['passed']==270 and record['skipped']==0 and record['failed']==0 and record['exit_code']==0
assert sha(ROOT/'logs/cpu-gate.log')==record['log_sha256']and sha(ROOT/'logs/cpu-gate-source.txt')==record['source_marker_sha256']
verify_monitor_cpu(approval)
for name,(b,mode)in payload.items():
 dst=ROOT/name
 if not dst.exists():
  with dst.open('xb')as f:f.write(b)
  dst.chmod(mode)
 assert sha(dst)==EXPECTED[name]
assert {x.name for x in (ROOT/'tools').iterdir()}==set(manifest['files'])
print(json.dumps({'status':'APPROVED_NEWROOT_TOOLS_DEPLOYED_NO_SUBMISSION_NO_GPU_ACCEPTANCE','source_commit':SOURCE,'files':51,'archive_sha256':ARCHIVE_SHA}),flush=True)
