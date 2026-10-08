import pathlib,json,os,signal,time
root=pathlib.Path('/data/shaoshitong/search-r1-rl/tree-grpo-retrieval');tag='cpu-encoder-fp32-http-fixed-20261003';run=root/'runs'/tag
report=json.loads((run/'real-encoder-difference.json').read_text());pid=report['pid'];proc=pathlib.Path('/proc')/str(pid)
expected=str(root/'tools/aux_retrieval_server.py')
cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode();assert expected in cmd and '--run '+tag in cmd,cmd
stat=(proc/'stat').read_text().rsplit(') ',1)[1].split();start=stat[19];ticks=os.sysconf('SC_CLK_TCK');cpu=(int(stat[11])+int(stat[12]))/ticks
before=time.time();os.kill(pid,signal.SIGTERM)
for _ in range(30):
 if not proc.exists():break
 time.sleep(.5)
forced=False
if proc.exists():
 stat2=(proc/'stat').read_text().rsplit(') ',1)[1].split()
 if stat2[19]==start:
  os.kill(pid,signal.SIGKILL);forced=True
samples=[json.loads(line) for line in (run/'gpu-samples.jsonl').read_text().splitlines()]
record={'scope':'stop own failed candidate only after real512 HTTP300s timeout; no production accepted','pid':pid,'proc_starttime':start,'term_epoch':before,'forced_own_pid_after15s':forced,'observer_stop_epoch':time.time(),'cpu_seconds_before_term':cpu,'observed_nvml_start_epoch':samples[0]['epoch'],'observed_gpu_sample_end_epoch':samples[-1]['epoch'],'observed_selected_gpu_span_hours':(samples[-1]['epoch']-samples[0]['epoch'])/3600,'gpu_peak_used_bytes':max(s['used_bytes'] for s in samples),'limitation':'Observer span is not exact total lease lifetime; use native cost.json if graceful finally exists'}
(root/'logs'/f'{tag}-own-stop.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
