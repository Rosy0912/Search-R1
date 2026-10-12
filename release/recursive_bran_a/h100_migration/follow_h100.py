"""Follow the running task without restarting jobs or acquiring GPU locks."""
import argparse,json,pathlib,subprocess,sys,time
p=argparse.ArgumentParser();p.add_argument('--socket',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parent;out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True);sock=a.socket
for index in range(2000):
    path=out/f'progress-{index:05d}.json'
    r=subprocess.run([sys.executable,str(root/'task_transport.py'),'--management-socket',sock,'--target','h100-11','--python-file',str(root/'live_progress.py'),'--output',str(path),'--timeout','45'])
    if r.returncode:
        print(json.dumps({'connection_failure':r.returncode,'training_action':'none'}),flush=True)
        binding=out/f'connection-{index:05d}.json'
        opened=subprocess.run([sys.executable,str(root/'open_task_transport.py'),str(binding),'20'])
        if opened.returncode==0:sock=json.loads(binding.read_text())['socket']
        time.sleep(5);continue
    d=json.loads(path.read_text());assert d['entry']['run_id']=='48528afa6b6443cfbb35328fd8b0ff05' and d['entry']['job']=='3387'
    receipts=d['completed_outer_receipts'];complete=max((v['outer'] for v in receipts),default=0)
    print(json.dumps({'time':d['time'],'owner':d['entry']['pid'],'process':d['process'],'complete':complete,
        'actual_writes':sum(v['writes'] for v in receipts),'latest_stage':d['latest_stage'],'error':d['error']}),flush=True)
    if d['error'] or d['process'] is None:
        print(json.dumps({'attention_required':True,'proof':str(path),'training_action':'none'}),flush=True);raise SystemExit(2)
    if complete>=10 and any(v['outer']>complete and v['event']=='trajectory_action' and v['count']>0 for v in d['counts']):
        print(json.dumps({'ten_rounds_and_next_generation_observed':True,'proof':str(path),'final_acceptance':'root inspection still required'}),flush=True);raise SystemExit(0)
    time.sleep(30)
raise SystemExit(3)
