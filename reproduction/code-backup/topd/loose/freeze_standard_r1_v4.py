import pathlib,json,hashlib,tarfile
root=pathlib.Path.cwd();d=root/'work/search-r1-32b-grpo100-tools-v4'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files={str(p.relative_to(d)):{'size':p.stat().st_size,'sha256':sha(p)} for p in sorted(d.rglob('*')) if p.is_file()}
assert len(files)==29 and not any('__pycache__' in p or p.endswith('.pyc') for p in files)
(d/'files.json').write_text(json.dumps({'files':files,'scope':'SOURCE_CPU_FREEZE_NOT_GPU_RESOURCE_NOT_TEACHER_NOT_TOPD','native_expected':36},indent=2,sort_keys=True)+'\n')
tar=d.with_suffix('.tar.gz')
with tarfile.open(tar,'w:gz') as t:
 for p in sorted(d.rglob('*')):
  if p.is_file():t.add(p,arcname=str(pathlib.Path(d.name)/p.relative_to(d)),recursive=False)
f=json.loads((root/'outputs/search-r1-32b-standard-v3-freeze.json').read_text());f.update(native_expected=36,local_result='35PASS_1_LINUX_ONLY_SKIP',local_log_sha256=sha(root/'outputs/search-r1-32b-standard-v4-local-cpu.log'),source_unchanged_from_v1=True,cache_budget_prior='0.6',cache_budget_current='0.4',failed_run=3159)
f['packages']['tools']={'directory':str(d),'archive':str(tar),'archive_sha256':sha(tar),'manifest_sha256':sha(d/'files.json'),'payload_count':29,'archive_members':30,'deployment_directory':'/dataset/shaoshitong/search-r1-rl/topd/tools/'+d.name}
(root/'outputs/search-r1-32b-standard-v4-freeze.json').write_text(json.dumps(f,indent=2)+'\n');print(json.dumps(f['packages']['tools'],indent=2))
