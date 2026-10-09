import subprocess,sys,shlex,json,pathlib
secret=input().strip()
code="""import pathlib,sys,subprocess,os,json,time
root=pathlib.Path('/data/shaoshitong/topd-hf-migration-20261009');token=sys.stdin.readline().strip()
if (root/'controller.json').exists():
 c=json.loads((root/'controller.json').read_text());assert not pathlib.Path('/proc/'+str(c['pid'])).exists(),'Already running'
env={k:v for k,v in os.environ.items() if k.lower() not in ['https_proxy','http_proxy','all_proxy','no_proxy']};env['HTTPS_PROXY']=env['HTTP_PROXY']='http://127.0.0.1:17894'
log=(root/'downloader.log').open('ab');p=subprocess.Popen(['python3','-u',str(root/'topd_hf_download_hkust3.py')],stdin=subprocess.PIPE,stdout=log,stderr=log,env=env,start_new_session=True);p.stdin.write((token+'\\n').encode());p.stdin.close();print(json.dumps({'pid':p.pid,'root':str(root),'epoch':time.time()}))
"""
r=subprocess.run(['ssh','-T','-B','en0','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes','hkust2','python3 -c '+shlex.quote(code)],input=secret+'\n',capture_output=True,text=True,timeout=20);print(r.stdout);print('exit',r.returncode)
pathlib.Path('outputs/hpc3-migration/hkust2-hf-launch.json').write_text(r.stdout)
