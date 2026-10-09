"""Control deployment via existing SSH aliases; credential only stdin/RAM."""
import json,pathlib,shlex,subprocess,sys
work=pathlib.Path(__file__).resolve().parent
plan=json.loads((work.parent/'outputs/hpc3-migration/hkust2-missing-assets-plan.json').read_text())
fresh=json.loads((work.parent/'outputs/hpc3-migration/remote-route-target-snapshot.json').read_text());accepted={x['group']:x for x in fresh['verified']};plan['records']=[x for x in plan['records'] if accepted.get(x['group'],{}).get('archive_sha256')!=x['archive_sha256']];assert plan['records']
for node,root,files in [('hkust3','/backup/shaoshitong/topd-hf-migration-20261009/direct-hpc3',{'topd_remote_missing_target.py':(work/'topd_remote_missing_target.py').read_text(),'missing-plan.json':json.dumps(plan)}),('hkust2','/data/shaoshitong/topd-hf-migration-20261009',{'topd_remote_missing_forward.py':(work/'topd_remote_missing_forward.py').read_text(),'pending-missing-plan.json':json.dumps(plan)})]:
 code="import pathlib,json,sys; r=pathlib.Path("+repr(root)+");d=json.load(sys.stdin);\nfor n,v in d.items():(r/n).write_text(v)\nprint('DEPLOYED')"
 p=subprocess.run(['ssh',node,'python3 -c '+shlex.quote(code)],input=json.dumps(files),text=True,capture_output=True,timeout=25);assert p.returncode==0,p.stderr[-200:];print(node,p.stdout.strip(),flush=True)
secret=sys.stdin.readline().strip();assert secret
code=r'''import pathlib,json,sys,subprocess,os,signal,time,fcntl
root=pathlib.Path('/data/shaoshitong/topd-hf-migration-20261009');token=sys.stdin.readline().strip();c=json.loads((root/'controller.json').read_text());pid=c['pid'];p=pathlib.Path('/proc/'+str(pid)+'/cmdline')
if p.exists():
 cmd=p.read_bytes();assert b'topd_hf_download_hkust3.py' in cmd;os.kill(pid,signal.SIGTERM)
 for _ in range(50):
  if not pathlib.Path('/proc/'+str(pid)).exists():break
  time.sleep(.1)
 assert not pathlib.Path('/proc/'+str(pid)).exists()
lock=(root/'downloader.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_UN)
(root/'plan-before-remote-missing.json').write_text((root/'plan.json').read_text());os.replace(root/'pending-missing-plan.json',root/'plan.json')
env={k:v for k,v in os.environ.items() if k.lower() not in ['https_proxy','http_proxy','all_proxy','no_proxy']};env['HTTPS_PROXY']=env['HTTP_PROXY']='http://127.0.0.1:17894'
log=(root/'downloader.log').open('ab');d=subprocess.Popen(['python3','-u',str(root/'topd_hf_download_hkust3.py')],stdin=subprocess.PIPE,stdout=log,stderr=log,env=env,start_new_session=True);d.stdin.write((token+'
').encode());d.stdin.close()
flog=(root/'missing-forward.log').open('ab');f=subprocess.Popen(['python3','-u',str(root/'topd_remote_missing_forward.py')],stdin=subprocess.DEVNULL,stdout=flog,stderr=flog,start_new_session=True)
r={'epoch':time.time(),'downloader_pid':d.pid,'forward_pid':f.pid,'replaced_downloader_pid':pid,'groups':[x['group'] for x in json.loads((root/'plan.json').read_text())['records']],'mac_data_used':False,'mac_proxy_for_HF':False,'partials_preserved':True};(root/'missing-launch.json').write_text(json.dumps(r));print(json.dumps(r))
'''
p=subprocess.run(['ssh','hkust2','python3 -c '+shlex.quote(code)],input=secret+'\n',capture_output=True,text=True,timeout=30);print(p.stdout);print('exit',p.returncode);assert p.returncode==0
(work.parent/'outputs/hpc3-migration/remote-missing-launch.json').write_text(p.stdout)
