python3 - <<'PY'
import pathlib,json,time,hashlib,base64,os
p=pathlib.Path('/var/log/slurm/slurmd.log');b=p.read_bytes();lines=b.splitlines(keepends=True);indices=[i for i,l in enumerate(lines) if b'3142' in l];lo=min(indices);hi=max(indices)+1;start=sum(map(len,lines[:lo]));end=start+sum(map(len,lines[lo:hi]));slice=b[start:end];st=p.stat()
print(json.dumps({'scope':'READ_ONLY_NATIVE_SLURMD_BATCH_TERMINAL_RAW_3142_NOT_SCONTROL_STATE','hostname':os.uname().nodename,'capture_epoch':time.time(),'source_path':str(p),'source_size':len(b),'source_sha256_at_capture':hashlib.sha256(b).hexdigest(),'source_mtime_ns':st.st_mtime_ns,'start_byte':start,'end_byte':end,'first_line':lo+1,'last_line':hi,'slice_sha256':hashlib.sha256(slice).hexdigest(),'slice_base64':base64.b64encode(slice).decode(),'slice_text':slice.decode()}))
PY
