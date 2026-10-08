"""Sync verified telemetry; acceptance remains the strict checker's decision."""
import json,re
from datetime import datetime,timezone
from pathlib import Path
from zoneinfo import ZoneInfo
base=Path(__file__).resolve().parent.parent
x=json.loads((base/'outputs/evidence/l-supervision-latest.json').read_text())
assert x['job']==2988 and x['run']=='cosearch-local-gpu-five-20261003-l'
steps=sorted({o['step'] for o in x.get('outer',[])})
p=base/'outputs/STATE.json';s=json.loads(p.read_text());s.update(last_supervision=x['checked_at'],actual_outer_updates=len(steps),ranker_partial_progress=x.get('rank',[]),latest_batch_progress=x.get('progress',[]),reload_status=x['reload_advance']);p.write_text(json.dumps(s,indent=2)+'\n')
p=base/'outputs/source-identity.json';s=json.loads(p.read_text());s['last_supervision']=dict(checked_at=x['checked_at'],run=x['run'],job=x['job'],outer_steps=steps,progress=x.get('progress',[]),rank=x.get('rank',[]),reload=x['reload_advance'],capture=x.get('capture'));s['active_worker_binding']=x.get('capture',s.get('active_worker_binding'));s['ranker_progress']=x.get('rank',[]);p.write_text(json.dumps(s,indent=2)+'\n')
stamp=datetime.fromtimestamp(x['checked_at'],timezone.utc).astimezone(ZoneInfo('Asia/Shanghai')).isoformat()
text='<!-- LIVE_L_SUPERVISION -->\n最近监督 '+stamp+'：'+x['slurm']+'，真实outer steps='+str(steps)+'；ranker累计='+json.dumps(x.get('rank',[]),ensure_ascii=False)+'；最近批进度='+json.dumps(x.get('progress',[]),ensure_ascii=False)+'。固定job2988/main PID2117449/startticks278751893/inode171125505采集绑定有效。加载推进状态='+x['reload_advance']['stdout'].strip()+'；每5分钟监督。诊断不计RL或验收。\n<!-- END_L_SUPERVISION -->\n'
p=base/'outputs/REPORT.md';s=p.read_text();s=re.sub(r'<!-- LIVE_L_SUPERVISION -->.*?<!-- END_L_SUPERVISION -->\n',lambda m:text,s,flags=re.S);p.write_text(s)
Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/cosearch/REPORT.md').write_text(s)
