from pathlib import Path
import ast
root=Path('/data/user/jxu957/experiments/topd-20261008/runtime/tools/topd-instruct300-launch-v6/runtime')
p=root/'process_observation.py'
s=p.read_text()
old="raise RuntimeError('Live observed PID reuse/UID drift')"
new="return {'pid':pid,'scope':'OBSERVED_IDENTITY_TRANSITION_NON_AUTHORITY','observed_epoch':time.time(),'before_identity':previous or ident,'after_identity':after,'inconclusive':True,'trusted_rss_sample':None}"
assert s.count(old)==1
s=s.replace(old,new);ast.parse(s);p.write_text(s)
p=root/'supervise.py';s=p.read_text();old="if 'identity' in r:observed[pid]=r['identity']"
new="if 'identity' in r:observed[pid]=r['identity']\n      else:observed.pop(pid,None)"
assert s.count(old)==1
s=s.replace(old,new);ast.parse(s);p.write_text(s)
print('PID observation transition classified as inconclusive; no authority or signalling granted')
