#!/bin/bash
#SBATCH --job-name=topd-io-cpu-1449
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/data/user/jxu957/experiments/topd-20261008/migration/cpu-io-probe-20261008-1449.log
export CUDA_VISIBLE_DEVICES=""
python3 - <<'PYCODE'
import pathlib,json,time,socket,hashlib,os
r=pathlib.Path('/data/user/jxu957/experiments/topd-20261008/migration');p=r/'cpu-io-probe-20261008-1449.bin';start=time.time();result={'started_epoch':start,'hostname':socket.gethostname(),'gpu_allocation':'none','role':'shared-storage IO diagnosis'}
try:
 h=hashlib.sha256();block=b'0123456789abcdef'*65536
 with p.open('wb') as f:
  for _ in range(256):f.write(block);h.update(block)
  f.flush();os.fsync(f.fileno())
 check=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):check.update(b)
 assert check.hexdigest()==h.hexdigest();result.update(status='WRITE_READ_SHA_PASS',bytes=p.stat().st_size,elapsed=time.time()-start,sha256=h.hexdigest());p.unlink()
except Exception as e:result.update(status='IO_PROBE_FAILED',error=repr(e),elapsed=time.time()-start)
(r/'cpu-io-probe-20261008-1449.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

PYCODE
