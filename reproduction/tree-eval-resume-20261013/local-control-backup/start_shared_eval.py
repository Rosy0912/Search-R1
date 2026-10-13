import importlib.util,json,shlex
from pathlib import Path
spec=importlib.util.spec_from_file_location('ssh',Path(__file__).with_name('ssh_retained18.py'));ssh=importlib.util.module_from_spec(spec);spec.loader.exec_module(ssh)
d=json.loads(Path('outputs/tree-eval-resume-20261013/SHARED_DEPLOYMENT.json').read_text());t=d['tools'];r='/dataset/shaoshitong/search-r1-rl/tree-grpo';e=r+'/runs/tree-branpo-fresh200-3360/output/eval-resume'
args=[r+'/env/bin/python',t+'/shared_dispatch_once.py','--unit','sst-tree-eval-shared-13a5388.service','--tools',t,'--prefix',e+'/bound-prefix-30880.json','--manifest',e+'/asset-manifest.json','--release',t+'/release.json','--score-output',e+'/scores','--journal',e+'/shared-repair3-dispatch-once.json','--review',t+'/SHARED_CODE_REVIEW.json','--repair-predecessor-unit','sst-tree-eval-shared-993109d.service','--repair-predecessor-journal',e+'/shared-repair2-dispatch-once.json']
x=ssh.run(shlex.join(args),timeout=120)
Path('outputs/tree-eval-resume-20261013/SHARED_DISPATCH.txt').write_text(x.stdout+'\n'+x.stderr+'\nexit '+str(x.returncode)+'\n');print(x.stdout,x.stderr,x.returncode)
