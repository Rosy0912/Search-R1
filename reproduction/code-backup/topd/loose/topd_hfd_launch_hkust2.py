"""Launch hfd with private token only in process RAM, no CLI/header files."""
import sys,subprocess,shlex,json,pathlib
secret=input().strip()
code=r'''import pathlib,sys,subprocess,os,json,time
root=pathlib.Path('/data/shaoshitong/topd-hfd-20261009');token=sys.stdin.readline().strip()
env={k:v for k,v in os.environ.items() if k.lower() not in ['https_proxy','http_proxy','all_proxy','no_proxy']};env['HTTPS_PROXY']=env['HTTP_PROXY']='http://127.0.0.1:18092';env['HF_TOKEN']=token
args=['bash',str(root/'hfd.sh'),'Sutoonq/TMP_MODEL','--hf_username','Sutoonq','--include','topd-migration-20261009/archives/wiki_corpus.tar.zst','-x','8','-j','1']
env['HF_ENDPOINT']='https://hf-mirror.com'
r=subprocess.run(args+['--local-dir',str(root/'mirror')],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=100)
(root/'mirror-test.json').write_text(json.dumps({'exit':r.returncode,'epoch':time.time(),'output':r.stdout.decode()[-1000:],'error':r.stderr.decode()[-1000:]}))
env['HF_ENDPOINT']='https://huggingface.co';log=(root/'hfd.log').open('ab');p=subprocess.Popen(args+['--local-dir',str(root/'official')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True);receipt={'pid':p.pid,'epoch':time.time(),'root':str(root),'endpoint':'https://huggingface.co','group':'wiki_corpus','connections':8,'token_persisted':False};(root/'controller.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
'''
p=subprocess.run(['ssh','-T','-B','en0','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes','hkust2','python3 -c '+shlex.quote(code)],input=secret+'\n',capture_output=True,text=True,timeout=120)
print(p.stdout);print('exit',p.returncode)
pathlib.Path('outputs/hpc3-migration/hfd-launch.json').write_text(p.stdout)
