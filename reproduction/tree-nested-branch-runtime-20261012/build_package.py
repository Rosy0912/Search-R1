"""Export committed code only; build fresh manifests without remote mutations."""
import argparse,hashlib,json,os,shutil,subprocess,tarfile,tempfile
from pathlib import Path
RUNTIME_REL='reproduction/tree-nested-branch-runtime-20261012'
ROOT='/dataset/shaoshitong/search-r1-rl/tree-grpo'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--instruction',required=True);a=p.parse_args()
 repo=a.repo.resolve();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
 if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip():raise ValueError('Commit all code/runtime before package')
 if a.output.exists():raise ValueError('No package overwrite')
 output=a.output.resolve()
 if output==repo or output.is_relative_to(repo):raise ValueError('Package must stay outside source repo')
 output.mkdir(parents=True);(output/'source/repo').mkdir(parents=True)
 archive=subprocess.check_output(['git','archive','--format=tar',commit],cwd=repo)
 with tempfile.TemporaryFile() as stream:
  stream.write(archive);stream.seek(0)
  with tarfile.open(fileobj=stream) as tar:
   for member in tar.getmembers():
    if member.issym() or member.islnk() or member.name.startswith('/') or '..' in Path(member.name).parts:raise ValueError('Unsafe code archive member')
   tar.extractall(output/'source/repo')
 runtime=output/'source/repo'/RUNTIME_REL
 for path in runtime.iterdir():
  if path.is_file():shutil.copy2(path,output/path.name)
  elif path.name=='config':shutil.copytree(path,output/path.name)
 m=json.loads((output/'manifest.template.json').read_text());m['code_commit']=commit
 m['independent_config']=dict(path='config/ppo_tree_nested_branch_fresh200.yaml',sha256=sha(output/'config/ppo_tree_nested_branch_fresh200.yaml'))
 remote=ROOT+'/instruct300/candidates/nested-branch-fresh200-'+commit
 intent=ROOT+'/instruct300/SUBMISSION_ONCE-nested-branch-fresh200-'+commit+'.json';m['dispatch']['once_intent']=intent
 authority=dict(status=m['execution_mode'],scope=m['scope'],code_commit=commit,human_instruction=a.instruction,same_instance_continuation=True,completed_outer_goal=200,final_evaluation_rows=51713,configured_prospective_thresholds=m['configured_prospective_thresholds'])
 for k in ('production_thresholds_accepted','independent_acceptance','training_accepted','original_sampler_actor_gate_accepted'):authority[k]=False
 (output/'human-authority.json').write_text(json.dumps(authority,indent=2)+'\n')
 m['human_training_authority']=dict(path=remote+'/human-authority.json',sha256=sha(output/'human-authority.json'))
 m['files']={str(f.relative_to(output)):sha(f) for f in sorted(output.rglob('*')) if f.is_file() and f.name not in ('manifest.json','package-build.json')}
 (output/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 # Build receipt is outside manifest/file inventory; records small immutable release inputs.
 receipt=dict(status='LOCAL_COMMITTED_CODE_PACKAGE_ONLY',source_commit=commit,remote_package=remote,manifest_sha256=sha(output/'manifest.json'),code_files=len(m['files']),actual_remote_executed=False,actual_submitted=False)
 (output/'package-build.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
