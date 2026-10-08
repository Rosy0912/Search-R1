import pathlib,json,hashlib,base64,tarfile,io,os
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-titlefix-r2-20261006');run=root/'runs/bapo-instruct300-hardware-3284'
assert run.is_dir() and not (run/'hardware.log').exists()
paths=sorted(p for p in run.rglob('*') if p.is_file() and not p.is_symlink())
paths += [root/'logs/slurm-hardware-3284.out',root/'logs/slurm-hardware-3284.err']
records=[]; stream=io.BytesIO()
with tarfile.open(fileobj=stream,mode='w:gz') as tf:
 for p in paths:
  a=p.stat();assert a.st_size<20000000
  raw=p.read_bytes();b=p.stat();assert (a.st_size,a.st_mtime_ns,a.st_ino)==(b.st_size,b.st_mtime_ns,b.st_ino)
  name=str(p.relative_to(root));h=hashlib.sha256(raw).hexdigest();records.append(dict(path=str(p),archive_name=name,bytes=len(raw),sha256=h))
  info=tarfile.TarInfo(name);info.size=len(raw);tf.addfile(info,io.BytesIO(raw))
raw=stream.getvalue();out=root/'logs/hardware3284-fault-raw.tar.gz'
with out.open('xb') as f:f.write(raw)
print(json.dumps(dict(status='ORIGINAL_CPU_FIXTURE_FAILURE_RAW_NOT_GPU_ACCEPTANCE',job_id=3284,records=records,archive_path=str(out),archive_bytes=len(raw),archive_sha256=hashlib.sha256(raw).hexdigest(),archive_base64=base64.b64encode(raw).decode())))
