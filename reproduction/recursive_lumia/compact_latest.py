"""Read-only latest physical updates and bounded log tails, including original A."""
import json
import math
import inspect
import subprocess
from pathlib import Path

def loss_status(metrics, outer, ranks, milestone_pre_present):
    if 'actor/pg_loss' in metrics:
        if metrics.get('training/actual_outer_updates') != outer or not math.isfinite(metrics['actor/pg_loss']):
            raise ValueError('Wrong-round or nonfinite aggregate policy loss')
        return dict(pg_loss=metrics['actor/pg_loss'], reported_pg_loss_finite=True,
                    loss_finite=True, loss_evidence='exact_round_console_aggregate',
                    summary_pending=False)
    # The optimizer receipt and sparse loss audits precede milestone evaluation;
    # the aggregate console line is emitted only after evaluation finishes.
    if (outer in (5,50,100,150,200) and milestone_pre_present and len(ranks)==8
            and {r['rank'] for r in ranks}==set(range(8))
            and all(r['observed_loss_scalars']>0 and r['all_observed_loss_finite'] is True for r in ranks)):
        return dict(pg_loss=None,reported_pg_loss_finite=None,loss_finite=True,
                    loss_evidence='all_eight_sparse_microbatch_loss_audits',summary_pending=True)
    raise ValueError('Current round lacks finite aggregate or complete milestone loss evidence')

CODE = r'''
import json,pathlib,sys,time,re,math
b=pathlib.Path('/home/jmzhang/Workspace/Recursive-Tree-GRPO-20261014-v1')
sys.path.insert(0,str(b/'sources/a520f27b8084c2644f5f183d9bcf1f1ee971e91a'))
from reproduction.recursive_lumia.inspect_run import verify_rank,sha
groups={}
for group,job,label in [('original','138650','original-gpu-r1'),('C1-root-only','138690','C1-resume11-L40S'),('C2-random-final','138704','parallel-C2-random-final'),('budget-20','138705','parallel-budget-20')]:
 run=b/'runs'/group;cfg=json.loads((run/'receipts/configuration.json').read_text())
 if group=='original':
  out=run/'output';assert str(json.loads((run/'receipts/train-start.json').read_text())['job'])==job
 else:
  entry=json.loads((run/'receipts/current-entry.json').read_text());out=pathlib.Path(entry['output'])
  assert str(entry['allocation'])==job and out.parent==run/'output'
  assert all(entry[k]==cfg[k] for k in ('run_id','source_commit','config_sha256','group'))
 identity=dict(job_id=job,run_id=cfg['run_id'],code_commit=cfg['source_commit'],config_sha256=cfg['config_sha256'],data_identity=cfg['data_identity'])
 path=sorted((out/'attempt-results').glob('attempt-*.json'))[-1];v=json.loads(path.read_text());outer=v['accepted_outer']
 ranks=v['all_rank_result']['ranks'];assert len(ranks)==8 and {r['rank'] for r in ranks}==set(range(8))
 verified=[verify_rank(r,run,identity,outer,out) for r in ranks]
 assert all(r['actual_adam_writes']==20 and r['total_adam_writes']==20*outer and r['all_observed_loss_finite'] is not False for r in verified)
 log=b/'logs'/(label+'-'+job+'.log');metrics={};last_event=None;tail=''
 for cap in (8*1024**2,32*1024**2):
  with log.open('rb') as f:
   offset=max(0,log.stat().st_size-cap);f.seek(offset);raw=f.read(cap)
  if offset:raw=raw.split(b'\n',1)[-1]
  tail=raw[:raw.rfind(b'\n')+1].decode(errors='replace')
  for line in tail.splitlines():
   if 'training/actual_outer_updates:' in line:
    m=re.search(r'(?:^|\s)step:(\d+) - ',line)
    if m and int(m.group(1))==outer:
     metrics={k:float(x) for k,x in re.findall(r'([A-Za-z0-9_/.-]+):([-+0-9.eE]+|nan|inf)',line[m.start():])}
   if 'TREE_TIMING=' in line or 'TREE_DRIVER_STAGE=' in line:
    marker='TREE_TIMING=' if 'TREE_TIMING=' in line else 'TREE_DRIVER_STAGE='
    try:event=json.JSONDecoder().raw_decode(line.split(marker,1)[1])[0]
    except ValueError:continue
    if event.get('run_id',identity['run_id'])==identity['run_id']:
     last_event={k:event[k] for k in ('event','stage','outer','unix_time','successful') if k in event}
  if 'actor/pg_loss' in metrics:break
 pre=run/'checkpoints/training_state'/('global_step_'+str(outer))/'manifest.json'
 loss=loss_status(metrics,outer,verified,pre.is_file())
 groups[group]=dict(outer=outer,identity=identity,actual_adam_writes_per_rank=20,ranks=verified,**loss,attempt_sha256=sha(path),receipt_mtime=path.stat().st_mtime,log_bytes=log.stat().st_size,log_mtime=log.stat().st_mtime,log_read_limit=cap,last_event=last_event,recent_error_lines=[l[-300:] for l in tail.splitlines()[-150:] if any(x in l for x in ('Traceback','Disk quota exceeded','OutOfMemoryError'))])
print(json.dumps(dict(at=time.time(),groups=groups,scope='Latest completed physical ranks and bounded console metrics; historical proofs retained separately.')))
'''

if __name__ == '__main__':
    r = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','lumia','python3 -'], input='import math\n'+inspect.getsource(loss_status)+'\n'+CODE, text=True, capture_output=True, timeout=55)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    value=json.loads(r.stdout)
    path=Path('lumia-compact-latest-audit.json')
    pending=path.with_suffix('.json.tmp');pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(path)
    print(json.dumps({g:dict(outer=v['outer'],loss_finite=v['loss_finite'],last_event=v['last_event']) for g,v in value['groups'].items()}))
