import sys,subprocess
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
script=sys.stdin.read()
p=subprocess.run(control.ssh('jump_h20-1')+['python3 -'],input=script,text=True,timeout=60)
raise SystemExit(p.returncode)
