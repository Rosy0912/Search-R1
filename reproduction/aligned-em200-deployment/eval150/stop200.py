import os,json,signal,time
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009');run=r/'runs/formal-aligned-em200-16c570e111f947e2b0026983df2f7938-3359';p=Path('/proc/1598631')
out={'human_instruction':'stop final200 inference; evaluate checkpoint150 on8GPUs larger batch','time':time.time()}
if p.exists():
 stat=(p/'stat').read_text();assert stat[stat.rfind(')')+2:].split()[19]=='632032726'
 out['cwd']=os.readlink(p/'cwd');out['cmdline']=(p/'cmdline').read_bytes().decode().replace('\0',' ')
 fd=os.pidfd_open(1598631);signal.pidfd_send_signal(fd,signal.SIGTERM);os.close(fd);out['signal']='own driver TERM'
cp=run/'checkpoints/actor/global_step_150';out['actor150_files']=[{'name':x.name,'size':x.stat().st_size}for x in cp.iterdir()]
out['root_files']=[x.name for x in r.iterdir()];(run/'human-stop-final200-20261012.json').write_text(json.dumps(out));print(json.dumps(out))
