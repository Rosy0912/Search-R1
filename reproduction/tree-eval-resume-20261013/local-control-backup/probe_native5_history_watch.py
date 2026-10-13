"""Small actual watcher and own-training resource observation."""
import argparse,json,shlex
from pathlib import Path
import ssh_retained18 as ssh
p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
code=r'''import json,subprocess,time
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');source='778b2b43dad90fc0c1c5d95d77c2326286011b73';run=root/'runs'/('tree-nested-branch-fresh200-not-slurm;systemd:'+source);unit='sst-tree-nested-branch-fresh200-778b2b43.service';watch='sst-tree-native5-history-watch-778b2b43.service';cg=Path('/sys/fs/cgroup/system.slice')/unit
path=run/'output/native-manifest-history/watcher-native5-778b2b43/state.json';state=json.loads(path.read_text()) if path.is_file() else None
print(json.dumps(dict(epoch=time.time(),state=state,state_path=str(path),watcher_systemd=subprocess.check_output(['systemctl','show',watch,'--property=MainPID,ActiveState,SubState,Result,ExecMainStatus,MemoryCurrent,CPUUsageNSec'],text=True),watcher_journal=subprocess.check_output(['journalctl','-u',watch,'-n','4','--no-pager','-o','cat'],text=True)[-1200:],training_systemd=subprocess.check_output(['systemctl','show',unit,'--property=MainPID,InvocationID,ActiveState,SubState,Result,CPUUsageNSec,MemoryCurrent,TasksCurrent'],text=True),own_cgroup={n:(cg/n).read_text() if (cg/n).is_file() else None for n in ('pids.current','pids.max','pids.events','memory.events','memory.peak')},archive_receipts=[p.name for p in (run/'output/native-manifest-history').glob('*.receipt.json')],native5_restore_receipt_created=(run/'output/nested-native5-loaded.json').is_file())))
'''
x=ssh.run('env CUDA_VISIBLE_DEVICES= python3 -c '+shlex.quote(code),timeout=60)
if x.returncode:raise RuntimeError(x.stderr[-2000:])
o=json.loads(x.stdout);a.output.write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(dict(epoch=o['epoch'],state=o['state'],own_cgroup=o['own_cgroup'],native5_loaded=o['native5_restore_receipt_created'],training=o['training_systemd'])))
