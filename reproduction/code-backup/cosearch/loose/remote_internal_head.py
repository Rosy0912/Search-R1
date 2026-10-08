"""Use the human-verified same-cluster head address; retain original management route."""
from pathlib import Path
p=Path(__file__).with_name('remote.py')
s=p.read_text()
old="inner = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'h100-16']"
new="inner = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'HostName=10.33.0.16', '-o', 'HostKeyAlias=100.108.173.4', '-o', 'StrictHostKeyChecking=yes', 'h100-16']"
assert s.count(old)==1
exec(compile(s.replace(old,new),str(p),'exec'),{'__name__':'__main__'})
