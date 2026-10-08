import pathlib,os,signal,time,json
root=pathlib.Path('/data/shaoshitong/search-r1-rl/tree-grpo-retrieval');run=root/'runs/fp32-index-http-20261003';pid=774407
expected=[str(root/'env/bin/python'),str(root/'tools/index_gpu_server.py'),'--run','fp32-index-http-20261003','--port','18082','--max-seconds','14400']
record=root/'logs/own-index-774407-user-stop.json';assert not record.exists()
provenance=json.loads((run/'provenance.json').read_text());assert provenance['pid']==pid and provenance['run']==str(run) and provenance['gpu_uuid']=='GPU-c63282c6-1d05-224e-234b-2a8c794b1f04'
def identity():
 p=pathlib.Path('/proc')/str(pid)
 if not p.exists():return None
 cmd=p.joinpath('cmdline').read_bytes().rstrip(b'\0').split(b'\0');assert [x.decode() for x in cmd]==expected
 return p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]
start=identity();actions=[];began=time.time()
if start is not None:
 os.kill(pid,signal.SIGTERM);actions.append('TERM_OWN_VERIFIED_PID')
 for _ in range(30):
  time.sleep(1)
  if not pathlib.Path('/proc',str(pid)).exists():break
 if pathlib.Path('/proc',str(pid)).exists():
  assert identity()==start
  os.kill(pid,signal.SIGKILL);actions.append('KILL_ONLY_SAME_OWN_PID_AFTER_30S')
cost=json.loads((run/'cost.json').read_text()) if (run/'cost.json').is_file() else None
out=dict(reason='User stopped cross-cluster retrieval architecture',pid=pid,start_time_ticks=start,start_epoch=began,end_epoch=time.time(),actions=actions,cost=cost,complete_lifetime_cost_available=cost is not None)
with record.open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))
