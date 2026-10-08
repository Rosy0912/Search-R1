"""Strict numeric logger parsing with two observed complete Ray trailing warning frames."""
import hashlib,math,re
NUMBER=r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
WARNING_BODY='FSDP.state_dict_type() and FSDP.set_state_dict_type() are being deprecated. Please use APIs, get_state_dict() and set_state_dict(), which can support different parallelisms, FSDP1, FSDP2, DDP. API doc: https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict.get_state_dict .Tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html .'
WARNING_PATH='/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py:689: FutureWarning: '
FRAME=re.compile(r'\(WorkerDict pid=([1-9][0-9]*)\) '+re.escape(WARNING_PATH+WARNING_BODY)+r'\Z')
CPU_AMP_BODY="`torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead."
CPU_AMP_PATH='/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/utils/checkpoint.py:1399: FutureWarning: '
CPU_AMP_FRAME=re.compile(r'\(WorkerDict pid=([1-9][0-9]*)\) '+re.escape(CPU_AMP_PATH+CPU_AMP_BODY)+r'\Z')
OBSERVED_FRAMES=(('observed_complete_trailing_ray_fsdp_futurewarning',FRAME),('observed_complete_trailing_ray_cpuamp_futurewarning',CPU_AMP_FRAME))
def parse_metric_summary(line):
 original=line
 line=re.sub(r'\x1b\[[0-9;]*m','',line)
 m=re.search(r'(?:^|\s)step:(\d+) - ',line)
 assert m,'Missing numeric logger step header'
 payload=line[m.end():];events=[]
 matches=[(kind,match) for kind,pattern in OBSERVED_FRAMES for match in [pattern.search(payload)] if match]
 assert len(matches)<=1,'Ambiguous trailing warning frame'
 if matches:
  kind,frame=matches[0]
  prefix=payload[:frame.start()]
  assert re.search(r'(?:^| - )timing_per_token_ms/gen:'+NUMBER+r'\Z',prefix),'Ray warning attached to incomplete/unexpected metric position'
  suffix=payload[frame.start():]
  events.append({'kind':kind,'step':int(m[1]),'worker_pid':int(frame[1]),'frame_sha256_ansi_removed':hashlib.sha256(suffix.encode()).hexdigest(),'original_physical_line_sha256':hashlib.sha256(original.encode()).hexdigest()})
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
