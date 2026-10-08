from pathlib import Path
import ast,json,hashlib,tarfile,io,gzip
b=Path(__file__).resolve().parent;old=b.parent/'e1-300-precision-oracle-r5';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();approval=json.loads((b/'launch-approval.json').read_text());manifest=json.loads((b/'tools-manifest.json').read_text());ROOT=approval['root'];SOURCE=approval['source_commit']
assert ROOT=='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-monitor-r6-20261004'and SOURCE=='733097feeae371b4efd8fd48940c46ff36da839b'
assert h(b/'tools-manifest.json')==approval['tools_manifest_sha256'];assert len(manifest['files'])==21 and len(approval['extra_helpers_sha256'])==27
report=b.parent.parent/'outputs/e1-300/INDEPENDENT-MONITOR-R6-LAUNCH-TOOLS-REVIEW.md';assert h(report)==approval['independent_review']['sha256']
worker={**{'tools/'+n:b/'tools'/n for n in manifest['files']},**{'evidence-tools/'+n:b/n for n in approval['extra_helpers_sha256']},'logs/tools-manifest.json':b/'tools-manifest.json','logs/launch-approval.json':b/'launch-approval.json','logs/'+report.name:report}
for n,p in worker.items():
 if n.startswith('tools/'):assert h(p)==manifest['files'][p.name]
 if n.startswith('evidence-tools/'):assert h(p)==approval['extra_helpers_sha256'][p.name]
assert len(worker)==51
headbind={n:h(b/'tools'/n) for n in ('submit.sh','hardware.sbatch','formal300.sbatch')};headbind['watch-terminal.py']=h(b/'watch-terminal.py');headbind.update(source_commit=SOURCE,root=ROOT,tools_manifest_sha256=h(b/'tools-manifest.json'),launch_approval_sha256=h(b/'launch-approval.json'),independent_review_sha256=h(report));(b/'approved-head-launch-hashes.json').write_text(json.dumps(headbind,indent=2)+'\n')
head={**{n:b/'tools'/n for n in ('submit.sh','hardware.sbatch','formal300.sbatch')},'watch-terminal.py':b/'watch-terminal.py','approved-launch-hashes.json':b/'approved-head-launch-hashes.json'}
def package(name,paths):
 p=b/name;assert not p.exists()
 with p.open('xb')as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0)as gz:
   with tarfile.open(fileobj=gz,mode='w')as t:
    for name,path in sorted(paths.items()):
     data=path.read_bytes();m=tarfile.TarInfo(name);m.size=len(data);m.mode=0o755 if path.suffix in ('.sh','.sbatch') else 0o644;m.mtime=0;t.addfile(m,io.BytesIO(data))
 return {'archive':p.name,'sha256':h(p),'bytes':p.stat().st_size,'files':{n:h(p)for n,p in sorted(paths.items())}}
ws=package('approved-launch-deploy.tar.gz',worker);hs=package('approved-head-deploy.tar.gz',head)
monitor='''\ndef verify_monitor_cpu(approval):
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
'''
for name,spec in [('deploy-approved-tools.py',ws),('deploy-approved-head.py',hs)]:
 s=(old/name).read_text().replace('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-precision-r5-20261004',ROOT)
 s='\n'.join(('EXPECTED='+repr(spec['files']))if line.startswith('EXPECTED=')else('ARCHIVE_SHA='+repr(spec['sha256']))if line.startswith('ARCHIVE_SHA=')else line for line in s.splitlines())+'\n'
 if name=='deploy-approved-tools.py':
  s=s.replace('len(payload)==49','len(payload)==51').replace("'files':49","'files':51").replace('EXPECTED=',monitor+'\nEXPECTED=',1)
  s=s.replace('for name,(b,mode)in payload.items():\n dst=ROOT/name\n if not dst.exists():','verify_monitor_cpu(approval)\nfor name,(b,mode)in payload.items():\n dst=ROOT/name\n if not dst.exists():')
 if name=='deploy-approved-tools.py':
  s=s.replace("assert approval['root']==str(ROOT)and approval['source_commit']==SOURCE and manifest['source_commit']==SOURCE","assert approval['root']==str(ROOT)and approval['source_commit']==SOURCE and manifest['source_commit']==SOURCE and manifest['root']==str(ROOT)\nassert approval['tools_sha256']==manifest['files']\nassert {'tools/'+n for n in manifest['files']}=={n for n in EXPECTED if n.startswith('tools/')}\nassert {'evidence-tools/'+n for n in approval['extra_helpers_sha256']}=={n for n in EXPECTED if n.startswith('evidence-tools/')}")
 p=b/name;assert not p.exists();p.write_text(s);ast.parse(s)
result={'status':'AUTHOR_APPROVED_PACKAGE_BINDINGS_REQUIRES_NONAUTHOR_DEPLOYMENT_REVIEW_NOT_GPU_ACCEPTANCE','root':ROOT,'source_commit':SOURCE,'tools_manifest_sha256':h(b/'tools-manifest.json'),'launch_approval_sha256':h(b/'launch-approval.json'),'independent_review_sha256':h(report),'worker':ws,'head':hs,'helper_sha256':{n:h(b/n)for n in ('deploy-approved-tools.py','deploy-approved-head.py')},'no_fixed_files_modified':True}
(b/'approved-deployment-packages.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ('worker','head')},indent=2));print('worker',ws['sha256']);print('head',hs['sha256'])
