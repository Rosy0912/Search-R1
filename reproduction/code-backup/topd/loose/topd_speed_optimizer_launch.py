"""Start one locked remote speed search without Mac data or proxy traffic."""
import base64,json,pathlib,shlex,subprocess,sys
data=base64.b64encode(pathlib.Path('work/topd_remote_speed_optimizer.py').read_bytes()).decode()
code=r'''
import pathlib,json,sys,subprocess,base64,time,os,fcntl
token=sys.stdin.readline().strip();assert token;r=pathlib.Path('/data/shaoshitong/topd-hf-route-probes-20261009')
lock=(r/'speed-optimizer.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
script=r/'topd_remote_speed_optimizer.py';script.write_bytes(base64.b64decode(DATA));fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy','no_proxy')}
p=subprocess.Popen(['python3',str(script)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True,env=env);p.stdin.write((token+'\n').encode());p.stdin.close();time.sleep(.5);assert p.poll() is None,'Optimizer startup failed';print(json.dumps({'pid':p.pid,'epoch':time.time(),'mac_used':False,'threshold_MB_s':5,'two_complete_32MiB_samples_required':True,'baseline_comparison_before_handoff':True}))
'''.replace('DATA',repr(data))
token=sys.stdin.readline().strip();p=subprocess.run(['ssh','-T','-B','en0','-b','10.4.162.158','-oProxyCommand=none','-oProxyJump=none','-oConnectTimeout=8','-oBatchMode=yes','hkust2','python3 -c '+shlex.quote(code)],input=token+'\n',capture_output=True,text=True,timeout=15)
out=json.loads(p.stdout) if p.returncode==0 else {'error':'optimizer_start_failed','exit':p.returncode}
pathlib.Path('outputs/hpc3-migration/speed-optimizer-launch.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
