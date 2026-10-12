import subprocess,json,pathlib,sys
W=pathlib.Path(__file__).resolve().parent;scope=sys.argv[1];remote='/home/lichen/h100-cleanup-20261012-'+scope
plan=json.loads((W/('selected.json' if scope=='tree-grpo' else 'logs-selected.json' if scope=='logs' else 'opsd-selected.json')).read_text());plan={'repo':'Sutoonq/TMP_MODEL','records':plan['records']}
ssh=['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ProxyCommand=none','-o','ProxyJump=none','mgmt-host']
subprocess.run(ssh+['mkdir -p '+remote],check=True)
for name,data in [('uploader.py',(W/'uploader.py').read_bytes()),('plan.json',json.dumps(plan).encode())]:subprocess.run(ssh+['cat > '+remote+'/'+name],input=data,check=True)
token=pathlib.Path('/Users/shaoshitong/.cache/huggingface/token').read_bytes().strip()+b'\n'
p=subprocess.Popen(ssh+['env -u LD_LIBRARY_PATH python3 '+remote+'/uploader.py'],stdin=subprocess.PIPE,stdout=(W/(scope+'-upload.log')).open('wb'),stderr=(W/(scope+'-upload.stderr')).open('wb'));p.communicate(token);print(json.dumps({'scope':scope,'exit_code':p.returncode}));sys.exit(p.returncode)
