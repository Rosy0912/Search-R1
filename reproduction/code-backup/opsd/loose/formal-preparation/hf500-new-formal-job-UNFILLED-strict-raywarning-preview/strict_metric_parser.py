"""Strict numeric logger parsing with one observed complete Ray trailing warning frame."""
import hashlib,math,re
NUMBER=r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
WARNING_BODY='FSDP.state_dict_type() and FSDP.set_state_dict_type() are being deprecated. Please use APIs, get_state_dict() and set_state_dict(), which can support different parallelisms, FSDP1, FSDP2, DDP. API doc: https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict .Tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html .'
WARNING_PATH='/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py:689: FutureWarning: '
FRAME=re.compile(r'\(WorkerDict pid=([1-9][0-9]*)\) '+re.escape(WARNING_PATH+WARNING_BODY)+r'\Z')
def parse_metric_summary(line):
 original=line
 line=re.sub(r'\x1b\[[0-9;]*m','',line)
 m=re.search(r'(?:^|\s)step:(\d+) - ',line)
 assert m,'Missing numeric logger step header'
 payload=line[m.end():];events=[]
 frame=FRAME.search(payload)
 if frame:
  prefix=payload[:frame.start()]
  assert re.search(r'(?:^| - )timing_per_token_ms/gen:'+NUMBER+r'\Z',prefix),'Ray warning attached to incomplete/unexpected metric position'
  suffix=payload[frame.start():]
  events.append({'kind':'observed_complete_trailing_ray_fsdp_futurewarning','step':int(m[1]),'worker_pid':int(frame[1]),'frame_sha256_ansi_removed':hashlib.sha256(suffix.encode()).hexdigest(),'original_physical_line_sha256':hashlib.sha256(original.encode()).hexdigest()})
  payload=prefix
 metrics={}
 for item in payload.split(' - '):
  pair=re.fullmatch(r'([A-Za-z_][A-Za-z0-9_./-]*):('+NUMBER+r')',item)
  assert pair,('Malformed numeric logger metric',item)
  key=pair[1];assert key not in metrics,('Duplicate numeric logger metric',key)
  value=float(pair[2]);assert math.isfinite(value),('Nonfinite numeric logger metric',key)
  metrics[key]=value
 assert metrics,'Empty numeric logger summary'
 return int(m[1]),metrics,events
