"""Reject existing compute processes on specified physical devices only."""
import csv
import io
import subprocess
import sys

def query(fields,kind):
    text=subprocess.check_output(['nvidia-smi','--query-'+kind+'='+fields,'--format=csv,noheader'],text=True)
    return list(csv.reader(io.StringIO(text)))

indices=set(sys.argv[1:])
assert indices and indices<={'0','1','2','3','5','6'}, 'GPU4 and GPU7 are reserved'
selected={row[1].strip() for row in query('index,uuid','gpu') if row[0].strip() in indices}
assert len(selected)==len(indices), 'Physical device missing'
busy=[row for row in query('gpu_uuid,pid','compute-apps') if row[0].strip() in selected]
assert not busy, 'Selected GPU has another process: '+repr(busy)
print('DEVICE_SCOPED_IDLE_OK',sorted(indices),flush=True)
