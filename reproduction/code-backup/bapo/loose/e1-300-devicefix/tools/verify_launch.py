import pathlib,json,hashlib,subprocess,sys
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-devicefix-20261004')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def validate_cpu_source_marker(root, source, record):
 marker=root/'logs/cpu-gate-source.txt'
 assert marker.is_file()and not marker.is_symlink()and marker.resolve()==marker
 assert marker.read_text().strip()==source=='733097feeae371b4efd8fd48940c46ff36da839b'
 assert record['source_commit']==source and record['source_marker_sha256']==sha(marker)
 return sha(marker)
phase=sys.argv[1];assert phase in ('hardware','formal300')
assert root.is_dir()and not root.is_symlink()and (root/'tools').is_dir()and not (root/'tools').is_symlink()
source=(root/'source-expected.txt').read_text().strip();assert source=='733097feeae371b4efd8fd48940c46ff36da839b';r=json.loads((root/'logs/launch-approval.json').read_text())
assert r['status']=='PASS_INDEPENDENT_E1_LAUNCH_REVIEW'and r['source_commit']==source and r['root']==str(root)
assert subprocess.check_output(['git','-C',str(root/'repo'),'rev-parse','HEAD'],text=True).strip()==source
assert not subprocess.check_output(['git','-C',str(root/'repo'),'status','--porcelain'],text=True)
assert sha(root/'logs/tools-manifest.json')==r['tools_manifest_sha256']
m=json.loads((root/'logs/tools-manifest.json').read_text());assert m['source_commit']==source
files=[p for p in (root/'tools').iterdir()if p.is_file()]
assert len(files)==21
assert all(not p.is_symlink()and p.resolve().parent==root/'tools'for p in files)
actual={p.name:sha(p)for p in files};assert actual==m['files']
assert sha(root/'repo/verl/trainer/config/ppo_bapo_e1_300.yaml')==r['config_yaml_sha256']
assert sha(root/'common-data/manifest.json')==r['data_manifest_sha256']=='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
assert sha(root/'assets/qwen2.5-3b-instruct-aa8e725/VERIFIED-MODEL-MANIFEST.json')==r['model_manifest_sha256']
assert sha(root/'logs/cpu-gate.json')==r['cpu_gate_record_sha256']
c=json.loads((root/'logs/cpu-gate.json').read_text());assert c['source_commit']==source and c['exit_code']==0 and c['failed']==0 and c['passed']==270
validate_cpu_source_marker(root,source,c)
assert c['skipped']==0 and c['scope']=='ACTUAL_TARGET_LINUX_CPU_NOT_GPU_ACCEPTANCE'
assert c['sys_prefix']==str(root/'env')
assert c['test_launcher_sha256']==sha(root/'tools/cpu-tests.py')
assert sha(root/'logs/cpu-gate.log')==c['log_sha256']
review=pathlib.Path(r['independent_review']['path']);assert review.is_file()and root in review.resolve().parents and sha(review)==r['independent_review']['sha256']
print('NEW_INSTRUCT300_FIXED_LAUNCH_AND_ACTUAL_CPU_GATE_PASS',phase,flush=True)
