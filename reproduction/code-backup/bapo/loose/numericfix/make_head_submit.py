from pathlib import Path
import base64
records={n:base64.b64encode((Path('work/numericfix/remote')/n).read_bytes()).decode() for n in ['parity-gpu64.sbatch','submit-hardware.sh']}
s='''import pathlib,base64,subprocess
root=pathlib.Path('/dev/shm/bapo-rewardfix-numeric-scheduler-20261004')
root.mkdir(exist_ok=True)
assert not root.is_symlink()
for name,data in RECORDS.items():
 p=root/name;b=base64.b64decode(data)
 if p.exists():assert p.read_bytes()==b,'Existing launch changed'
 else:p.write_bytes(b)
subprocess.run(['bash',str(root/'submit-hardware.sh')],check=True)
'''.replace('RECORDS',repr(records))
Path('work/numericfix/head-submit.py').write_text(s)
