"""Existing allocation CPU-only compatibility tests; not production first5 inspection."""
import argparse,hashlib,json,os,pathlib,re,socket,subprocess,sys,time,types,unittest
P=pathlib.Path
CANDIDATE='96fe900ce618bce4b8a11eee42015cd32fe8b29db3fb676a957b841fbbe6048d'
SOURCE='89beac273f783d8e48d7d3e34da040f753514d90'
FIXTURE_SHA='8c0c1847654fa818ed4c15c20932ec96f7d90e599538c5f2ce596d3392b8ab86'
FULL_STATE_SHA='b7239eca041d317c1c19738c74ff7659fff4bde61df3267d25ea5dcddda6ffe6'
BASE_MANIFEST_SHA='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'

def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):
 path=P(path);require(path.is_absolute() and path.resolve(strict=True)==path and path.is_file(),'Physical file required')
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def validate_tool_bundle(directory):
 directory=P(directory);require(directory.is_absolute() and directory.resolve(strict=True)==directory and directory.stat().st_uid==os.getuid(),'Own physical tool directory required')
 manifest=directory/'candidate-manifest.json';require(sha(manifest)==CANDIDATE,'Frozen V3 candidate SHA differs')
 m=json.loads(manifest.read_bytes());require(m['version']==3 and m['scope']=='CPU_SOFTWARE_CANDIDATE_NOT_ACTUAL_FIRST5_ACCEPTANCE','Wrong candidate scope')
 require(set(m['files'])=={'inspect_payloads.py','test_inspector.py','README.md','cpu-tests.txt','v2-to-v3.diff'},'Tool bundle file inventory differs')
 for name,rec in m['files'].items():require(P(name).name==name and sha(directory/name)==rec['sha256'],'Frozen tool bytes differ')
 require(not list(directory.rglob('*.pyc')),'No old tool bytecode allowed')
 return m

def main():
 p=argparse.ArgumentParser();p.add_argument('--tools',required=True);p.add_argument('--repo',required=True);p.add_argument('--fixture',required=True);p.add_argument('--base-manifest',required=True);p.add_argument('--actual-job',required=True);a=p.parse_args()
 require(a.actual_job.isdigit() and os.environ.get('SLURM_JOB_ID')==a.actual_job,'Actual existing allocation required')
 require(socket.gethostname().split('.')[0]=='h100-10' and os.getuid()==0,'Existing native node/UID differ')
 cgroup=P('/proc/self/cgroup').read_text();require(re.search(r'(?:^|/)job_'+re.escape(a.actual_job)+r'(?:/|$)',cgroup,re.M) is not None,'CPU step not in actual allocation cgroup')
 status=P('/proc/self/status').read_text();uids=re.search(r'^Uid:\s+(.*)',status,re.M);require(uids is not None and set(map(int,uids.group(1).split()))=={0},'CPU step four UIDs differ')
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('OMP_NUM_THREADS')=='1' and os.environ.get('PYTHONDONTWRITEBYTECODE')=='1','CPU-hidden single-thread no-pyc environment required')
 bundle=validate_tool_bundle(a.tools);repo=P(a.repo);fixture=P(a.fixture);base=P(a.base_manifest)
 require(repo.is_absolute() and repo.resolve(strict=True)==repo and repo.name=='repo-'+SOURCE[:12],'Actual frozen repo path differs')
 require(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==SOURCE and not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip(),'Actual fixed source not clean')
 require(sha(repo/'search_r1/baselines/full_state.py')==FULL_STATE_SHA and sha(fixture)==FIXTURE_SHA and sha(base)==BASE_MANIFEST_SHA,'Actual fixture/helper/official metadata differs')
 os.environ.update(FIRST5_TEST_REPO=str(repo),FIRST5_TEST_FIXTURE=str(fixture),FIRST5_TEST_BASE_MANIFEST=str(base))
 import torch
 require(torch.__version__=='2.4.0+cu121' and not torch.cuda.is_initialized(),'Native Torch2.4 CPU before tests required')
 test=P(a.tools)/'test_inspector.py';module=types.ModuleType('first5_native_cpu_tests');module.__file__=str(test)
 exec(compile(test.read_bytes(),str(test),'exec'),module.__dict__)
 suite=unittest.defaultTestLoader.loadTestsFromModule(module);require(suite.countTestCases()==42,'Actual test inventory differs')
 start=time.monotonic();result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(suite)
 require(not torch.cuda.is_initialized(),'Tests initialized CUDA')
 require(result.wasSuccessful() and result.testsRun==42,'CPU compatibility regression failed')
 validate_tool_bundle(a.tools)
 require(sha(repo/'search_r1/baselines/full_state.py')==FULL_STATE_SHA and sha(fixture)==FIXTURE_SHA and sha(base)==BASE_MANIFEST_SHA,'Actual input drifted during tests')
 require(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==SOURCE and not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip(),'Frozen source drifted during tests')
 print('FIRST5_V3_NATIVE_CPU_COMPAT_JSON='+json.dumps({'scope':'NATIVE_TORCH24_CPU_SOFTWARE_COMPATIBILITY_USING_OLD3190_TINY_FIXTURE_NOT_PRODUCTION_FIRST5','candidate_sha256':CANDIDATE,'inspector_sha256':bundle['files']['inspect_payloads.py']['sha256'],'source_commit':SOURCE,'actual_cpu_step_job_id':a.actual_job,'node':socket.gethostname(),'uid':os.getuid(),'pid':os.getpid(),'cgroup':cgroup,'torch_version':torch.__version__,'tests_passed':42,'seconds':time.monotonic()-start,'fixture_path':str(fixture),'fixture_sha256':FIXTURE_SHA,'full_state_sha256':FULL_STATE_SHA,'official_manifest_sha256':BASE_MANIFEST_SHA,'cuda_is_initialized':False,'cuda_visible_devices':'','passed_software_compatibility_only':True},sort_keys=True))
if __name__=='__main__':main()
