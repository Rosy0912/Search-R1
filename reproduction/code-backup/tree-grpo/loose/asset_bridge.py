import importlib.util, shlex, subprocess, threading
s=importlib.util.spec_from_file_location('routes','/Users/shaoshitong/Documents/remote_codex_control/control.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
source=m.ssh('jump_h20-1');source=source[:-1]+['-N','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-L','127.0.0.1:19992:127.0.0.1:19992',source[-1]]
a=subprocess.Popen(source)
port='19993';opts=['-T','-o','BatchMode=yes','-o','ExitOnForwardFailure=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-R',f'127.0.0.1:{port}:127.0.0.1:{port}']
worker='env -u LD_LIBRARY_PATH '+shlex.join(['ssh']+opts+['h100-9','printf TREE_ASSET_BRIDGE_READY\\n; sleep 86400'])
head='env -u LD_LIBRARY_PATH '+shlex.join(['ssh']+opts+['h100-16',worker])
outer=m.ssh('mgmt-host');outer=outer[:-1]+['-o','ExitOnForwardFailure=yes','-R',f'127.0.0.1:{port}:127.0.0.1:19992',outer[-1],head]
b=subprocess.Popen(outer)
try: b.wait()
finally:
 a.terminate();b.terminate();a.wait();b.wait()
