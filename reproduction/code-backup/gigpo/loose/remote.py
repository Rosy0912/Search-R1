import subprocess,sys,shlex
sys.path.insert(0,"/Users/shaoshitong/Documents/remote_codex_control")
import control
node=sys.argv[1]
script=sys.stdin.read()
cmd='env -u LD_LIBRARY_PATH ssh -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 '+shlex.quote(node)+' '+shlex.quote('bash -s')
mid='env -u LD_LIBRARY_PATH ssh -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 h100-16 '+shlex.quote(cmd)
raise SystemExit(subprocess.run(control.ssh('mgmt-host')+[mid],input=script,text=True).returncode)
