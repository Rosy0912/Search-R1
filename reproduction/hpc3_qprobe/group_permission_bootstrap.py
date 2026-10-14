"""Exact retained bootstrap for the human's explicit shared-group770 grant.

Executed once via the original owner's control bridge. Its saved process intent
prevents replay; preserve the earlier unsupported-ACL failure evidence separately.
"""
import os,json,time,subprocess
from pathlib import Path
B=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014');root=Path('/data/user/jxu957/experiments');home=root.parent
assert os.getuid()==21930 and home.stat().st_uid==21930 and root.stat().st_uid==21930
receipt=B/'preparation/group770-permission-intent.json';assert not receipt.exists(),'Inspect existing permission process; do not duplicate'
out=dict(authorisation='用户明确指定 chmod -R 770 /data/user/jxu957/experiments/；两个账号共同读写',epoch=time.time(),home_before=oct(home.stat().st_mode & 0o777),experiments_before=oct(root.stat().st_mode & 0o777),target_group=21919)
with receipt.open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
# Preserve home privacy while permitting group traversal into the authorized subtree.
os.chmod(home,(home.stat().st_mode & 0o7777)|0o010)
log=B/'preparation/group770-permission-process.log'
with log.open('xb') as f:
 p=subprocess.Popen(['chmod','-R','770',str(root)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
out.update(pid=p.pid,start_ticks=Path('/proc/'+str(p.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19],log=str(log),home_after=oct(home.stat().st_mode & 0o777))
with (B/'preparation/group770-permission-process.json').open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(out,ensure_ascii=False))
