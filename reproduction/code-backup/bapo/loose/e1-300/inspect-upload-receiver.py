import pathlib,json,os,time
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-20261004/assets/qwen2.5-3b-instruct-aa8e725');p=root/'model-00001-of-00002.safetensors.part'
def inspect():
 receivers=[];writers=[]
 for proc in pathlib.Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:cmd=(proc/'cmdline').read_bytes()
  except OSError:continue
  if str(root).encode()in cmd and b'REMOTE_OFFICIAL_FILE_VERIFIED'in cmd and b'sys.stdin.buffer.read'in cmd:
   receivers.append(int(proc.name))
  try:
   for fd in(proc/'fd').iterdir():
    try:
     if os.readlink(str(fd))==str(p):
      flags=int(next(x.split()[1]for x in(proc/'fdinfo'/fd.name).read_text().splitlines()if x.startswith('flags:')),8)
      if flags&os.O_ACCMODE in(1,2):writers.append({'pid':int(proc.name),'fd':int(fd.name)})
    except OSError:continue
  except OSError:continue
 st=p.stat();return dict(receiver_pids=receivers,write_fds=writers,bytes=st.st_size,mtime_ns=st.st_mtime_ns)
a=inspect();time.sleep(5);b=inspect();print(json.dumps({'before':a,'after':b,'same_partial':a['bytes']==b['bytes']and a['mtime_ns']==b['mtime_ns'],'no_receiver_or_writer':not(a['receiver_pids']or b['receiver_pids']or a['write_fds']or b['write_fds'])}),flush=True)
