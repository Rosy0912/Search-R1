"""Generate bounded, unbuffered shared-storage receiver with exact partial-write handling."""
import textwrap

def receiver_code(path):
 return textwrap.dedent('''\
 from pathlib import Path
 import sys,os,time
 p=Path(PATH);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('ab',buffering=0) as f:
  for chunk in iter(lambda:sys.stdin.buffer.read(4194304),b''):
   remaining=memoryview(chunk);failures=0
   while remaining:
    try:
     count=os.write(f.fileno(),remaining)
     if count<=0:raise RuntimeError('Zero-byte write')
     remaining=remaining[count:];failures=0
    except OSError as exc:
     if exc.errno!=5 or failures>=5:raise
     failures+=1
     print('Shared-storage EIO; bounded retry '+str(failures),file=sys.stderr,flush=True)
     time.sleep(2)
  for attempt in range(6):
   try:os.fsync(f.fileno());break
   except OSError as exc:
    if exc.errno!=5 or attempt==5:raise
    time.sleep(2)
 ''').replace('PATH',repr(path))
