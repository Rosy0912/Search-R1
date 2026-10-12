import pathlib,subprocess,json
W=pathlib.Path(__file__).resolve().parent;remote='/home/lichen/h100-cleanup-20261012-node10-controller';ssh=['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ProxyCommand=none','-o','ProxyJump=none','mgmt-host']
subprocess.run(ssh+['mkdir -p '+remote],check=True)
for name in ['cleanup_controller.py','delete_success.py']:subprocess.run(ssh+['cat > '+remote+'/'+name],input=(W/name).read_bytes(),check=True)
p=subprocess.Popen(ssh+['python3 '+remote+'/cleanup_controller.py'],stdin=subprocess.PIPE,stdout=(W/'controller.log').open('wb'),stderr=(W/'controller.stderr').open('wb'));p.communicate(pathlib.Path('/Users/shaoshitong/.cache/huggingface/token').read_bytes().strip()+b'\n');print(json.dumps({'exit_code':p.returncode}));raise SystemExit(p.returncode)
