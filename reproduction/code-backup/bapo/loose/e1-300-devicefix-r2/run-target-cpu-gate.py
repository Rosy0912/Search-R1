import pathlib,sys,subprocess,os,time,json,re,hashlib
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-devicefix-r2-20261004');source='733097feeae371b4efd8fd48940c46ff36da839b'
assert subprocess.check_output(['git','-C',str(r/'repo'),'rev-parse','HEAD'],text=True).strip()==source
assert not subprocess.check_output(['git','-C',str(r/'repo'),'status','--porcelain'],text=True)
marker=r/'logs/cpu-gate-source.txt'
with marker.open('x')as f:f.write(source+'\n')
marker_sha=hashlib.sha256(marker.read_bytes()).hexdigest()
log=r/'logs/cpu-gate.log';env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONPATH']=str(r/'repo');env.pop('LD_LIBRARY_PATH',None);start=time.monotonic()
with log.open('x')as f:p=subprocess.run([str(r/'env/bin/python'),str(r/'tools/cpu-tests.py')],cwd=r/'repo',env=env,stdout=f,stderr=subprocess.STDOUT)
raw=log.read_text();passed=re.findall(r'(\d+) passed',raw);skipped=re.findall(r'(\d+) skipped',raw);failed=re.findall(r'(\d+) failed',raw)
assert subprocess.check_output(['git','-C',str(r/'repo'),'rev-parse','HEAD'],text=True).strip()==source
assert not subprocess.check_output(['git','-C',str(r/'repo'),'status','--porcelain'],text=True)
def origins(marker):
 rows=[json.loads(line.split(marker,1)[1])for line in raw.splitlines()if marker in line]
 return rows[0]if len(rows)==1 else None
import_origins={'before':origins('CPU_IMPORT_ORIGINS_BEFORE_JSON='),'after':origins('CPU_IMPORT_ORIGINS_AFTER_JSON=')}
record={'import_origins':import_origins,'source_commit':source,'source_marker_sha256':marker_sha,'exit_code':p.returncode,'passed':int(passed[-1])if passed else 0,'skipped':int(skipped[-1])if skipped else 0,'failed':int(failed[-1])if failed else 0,'wall_seconds':time.monotonic()-start,'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),'test_launcher_sha256':hashlib.sha256((r/'tools/cpu-tests.py').read_bytes()).hexdigest(),'sys_prefix':sys.prefix,'scope':'ACTUAL_TARGET_LINUX_CPU_NOT_GPU_ACCEPTANCE'}
(r/'logs/cpu-gate.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
sys.exit(p.returncode)
