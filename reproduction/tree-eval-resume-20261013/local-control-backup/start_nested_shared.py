"""One authorized continuous fresh200 dispatch on the original asset-local18."""
import importlib.util,json,shlex
from pathlib import Path
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('ssh',HERE/'ssh_retained18.py')
ssh=importlib.util.module_from_spec(spec);spec.loader.exec_module(ssh)
d=json.loads(Path('outputs/tree-nested-branch-handoff-20261013/DELTA_DEPLOYMENT.json').read_text())
script=json.loads(Path('outputs/tree-nested-branch-handoff-20261013/RENDERED_DISPATCH.json').read_text())
root='/dataset/shaoshitong/search-r1-rl/tree-grpo';pkg=d['package']
args=['env','CUDA_VISIBLE_DEVICES=',root+'/env/bin/python',pkg+'/submit_shared_once.py','--package',pkg,'--manifest-sha',d['manifest_sha256'],'--release-path',pkg+'/release.json','--release-sha',d['release_sha256'],'--script',script['path'],'--script-sha',script['sha256']]
Path('outputs/tree-nested-branch-handoff-20261013/LOCAL_SUBMISSION_INTENT.json').write_text(json.dumps(dict(status='ONE_CALL_STARTED_RECONCILE_BEFORE_ANY_RETRY',source_commit=d['source_commit'],args=args,unit='sst-tree-nested-branch-fresh200-'+d['source_commit'][:8]+'.service'),indent=2)+'\n')
r=ssh.run(shlex.join(args),timeout=900)
Path('outputs/tree-nested-branch-handoff-20261013/SHARED_SUBMISSION_RESULT.txt').write_text(r.stdout+'\n'+r.stderr+'\nexit '+str(r.returncode)+'\n')
if r.returncode:raise RuntimeError(r.stdout[-4000:]+r.stderr[-4000:])
receipt=json.loads(r.stdout)
Path('outputs/tree-nested-branch-handoff-20261013/SHARED_SUBMISSION_RESULT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:receipt[k] for k in ['status','unit','job_id','run_id','start_epoch','end_epoch','physical_gpus','slurm_reserved_gpus']}))
