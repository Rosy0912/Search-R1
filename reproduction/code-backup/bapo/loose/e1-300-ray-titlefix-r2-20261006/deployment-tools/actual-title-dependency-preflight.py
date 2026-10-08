"""CPU-only real Ray-bundled setproctitle/proc proof; no Ray.init or signals."""
import argparse,base64,hashlib,importlib.util,inspect,json,os,pathlib,select,subprocess,sys,time
ROOT='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-titlefix-r2-20261006'
SOURCE='3f85dabc7aa05bc45ec7af015d65d9fa4a4c17e4'
EXT_SHA='9429a267fba07b2ec523ae906878e154dbe053065c4e89c7e450037866c2557b'
def need(v,msg):
 if not v:raise ValueError(msg)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def ordinary(p):
 p=pathlib.Path(p);need(p.is_file()and not p.is_symlink(),'Ordinary original required');return p

def dependency():
 import ray,setproctitle,ray._private.worker as worker
 p=pathlib.Path(setproctitle.__file__).resolve();expected=(pathlib.Path(ray.__file__).resolve().parent/'thirdparty_files'/'setproctitle.cpython-310-x86_64-linux-gnu.so').resolve()
 need(ray.__version__=='2.40.0'and p==expected and p.stat().st_size==69104 and sha(p)==EXT_SHA,'Existing Ray extension identity differs')
 need(worker.setproctitle.setproctitle is setproctitle.setproctitle,'Ray worker title function differs')
 return setproctitle,dict(ray_version=ray.__version__,ray_init_path=str(pathlib.Path(ray.__file__).resolve()),ray_init_sha256=sha(ray.__file__),extension_path=str(p),extension_bytes=p.stat().st_size,extension_sha256=sha(p),function_module=setproctitle.setproctitle.__module__,function_name=setproctitle.setproctitle.__name__,same_as_ray_worker=True,function_builtin=inspect.isbuiltin(setproctitle.setproctitle),worker_title_source=inspect.getsource(worker._changeproctitle))

def proc(pid):
 p=pathlib.Path('/proc')/str(pid);stat=(p/'stat').read_text();tail=stat[stat.rfind(')')+2:].split();exe=(p/'exe').resolve();s=exe.stat();raw=(p/'cmdline').read_bytes()
 return dict(pid=pid,start_ticks=int(tail[19]),executable=dict(resolved_path=str(exe),device=s.st_dev,inode=s.st_ino,sha256=sha(exe)),cmdline_base64=base64.b64encode(raw).decode(),command=raw.decode().split('\0'),uid=p.stat().st_uid)
def validate_snapshot(before,after,title):
 for k in ('pid','start_ticks','executable','uid'):need(before[k]==after[k],'Child identity changed '+k)
 raw=base64.b64decode(after['cmdline_base64'],validate=True);need(raw==('\0'.join(after['command'])).encode(),'Raw command differs');need(after['command'][0]==title and not any(after['command'][1:]),'Real title or NUL padding differs')

def child():
 need(sys.platform=='linux'and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only Linux child required');title,dep=dependency();print(json.dumps(dict(type='ready',pid=os.getpid(),dependency=dep)),flush=True);deadline=time.monotonic()+90
 while time.monotonic()<deadline:
  if not select.select([sys.stdin],[],[],min(1,max(0,deadline-time.monotonic())))[0]:continue
  line=sys.stdin.readline()
  if not line:return
  request=json.loads(line)
  if request=={'stop':True}:return
  value=request['title'];title.setproctitle(value);print(json.dumps(dict(type='ack',title=value,sequence=request['sequence'],snapshot=proc(os.getpid()))),flush=True)
 raise TimeoutError('Own child natural bounded deadline')
def read(child):
 need(select.select([child.stdout],[],[],10)[0],'Child bounded acknowledgement missing');line=child.stdout.readline();need(line,'Own child exited before acknowledgement');return json.loads(line)
def controller(output):
 need(sys.platform=='linux'and pathlib.Path(sys.prefix)==pathlib.Path(ROOT)/'env','Actual target Linux interpreter required')
 need(pathlib.Path(sys.executable).resolve()==(pathlib.Path(ROOT)/'env/bin/python').resolve(),'Actual env interpreter differs')
 tools=pathlib.Path(ROOT)/'tools';manifest_path=ordinary(pathlib.Path(ROOT)/'logs/tools-manifest.json');manifest=json.loads(manifest_path.read_text());need(manifest['source_commit']==SOURCE,'Manifest source differs')
 for name in ('actual-title-dependency-preflight.py','identity-linux-fixture.py','resource_watch.py'):
  p=ordinary(tools/name);need(sha(p)==manifest['files'][name],'Actual manifest tool differs '+name)
 need(pathlib.Path(__file__).resolve()==tools/'actual-title-dependency-preflight.py','Manifest-bound tool route required')
 actual=subprocess.check_output(['git','-C',ROOT+'/repo','rev-parse','HEAD'],text=True).strip();need(actual==SOURCE,'Actual fixed source differs')
 need(not subprocess.check_output(['git','-C',ROOT+'/repo','status','--porcelain'],text=True).strip(),'Source must remain clean')
 spec=importlib.util.spec_from_file_location('approved_watcher',tools/'resource_watch.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w);titles=sorted(w.RAY_APPROVED_TITLES);need(len(titles)==25,'Exact25 approved titles required')
 _,parent_dependency=dependency();out=pathlib.Path(output);need(out.parent==pathlib.Path(ROOT)/'logs'and not out.exists(),'Fresh newROOT logs output required');env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='');env.pop('PYTHONPATH',None)
 c=subprocess.Popen([sys.executable,'-B',str(tools/'actual-title-dependency-preflight.py'),'--child'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env);record=dict(scope='ACTUAL_LINUX_RAY_VENDORED_TITLE_CPU_ONLY_NOT_GPU_OR_TRAINING',root=ROOT,source_commit=SOURCE,sys_prefix=sys.prefix,tool_manifest_sha256=sha(manifest_path),fixture_sha256=manifest['files']['identity-linux-fixture.py'],self_sha256=sha(__file__),acks=[],signals_sent=0,ray_init_called=False,actual_gpu_accepted=False)
 try:
  ready=read(c);need(ready['type']=='ready'and ready['pid']==c.pid and ready['dependency']==parent_dependency,'Child ready identity differs');record['ready']=ready;baseline=proc(c.pid)
  for index,title in enumerate(titles):
   c.stdin.write(json.dumps(dict(sequence=index,title=title))+'\n');c.stdin.flush();ack=read(c);need(ack['type']=='ack'and ack['title']==title and ack['sequence']==index,'Actual acknowledgement differs');external=proc(c.pid);need(ack['snapshot']==external,'Child/proc snapshot differs');validate_snapshot(baseline,external,title);record['acks'].append(ack)
  _,after_dependency=dependency();need(after_dependency==parent_dependency,'Actual dependency changed during25 titles');record['dependency_after']=after_dependency
 finally:
  if c.poll()is None:
   try:c.stdin.write('{"stop":true}\n');c.stdin.flush()
   except BrokenPipeError:pass
  c.stdin.close();c.wait(timeout=95);stderr=c.stderr.read();record.update(child_exit_code=c.returncode,child_stderr=stderr)
 need(c.returncode==0 and stderr==''and len(record['acks'])==25,'Own child did not complete naturally');record['status']='PASS_ACTUAL_CPU25_TITLES_NOT_GPU_OR_PERMISSION';record['original_process']=baseline
 with out.open('x')as f:f.write(json.dumps(record,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(status=record['status'],path=str(out),sha256=sha(out),bytes=out.stat().st_size)))
def main():
 p=argparse.ArgumentParser();p.add_argument('--child',action='store_true');p.add_argument('--output');a=p.parse_args()
 if a.child:need(a.output is None,'No child output path');child()
 else:need(a.output,'Actual output required');controller(a.output)
if __name__=='__main__':main()
