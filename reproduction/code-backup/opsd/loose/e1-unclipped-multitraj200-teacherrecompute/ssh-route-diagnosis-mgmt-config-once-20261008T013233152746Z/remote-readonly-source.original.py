import json,os,socket,subprocess,re,stat
from pathlib import Path
r={"hostname":socket.gethostname(),"uid":os.getuid(),"ssh_auth_sock_present":bool(os.environ.get("SSH_AUTH_SOCK")),"aliases":{},"ssh_files":[]}
for alias in ("h100-16","h100-10"):
 p=subprocess.run(["env","-u","LD_LIBRARY_PATH","ssh","-G",alias],capture_output=True,text=True)
 fields=[]
 for line in p.stdout.splitlines():
  k=line.split(" ",1)[0]
  if k in {"hostname","user","port","identityfile","identityagent","identitiesonly","proxycommand","proxyjump","forwardagent","hostkeyalias","userknownhostsfile"}:
   fields.append(line if not re.search("sshpass|password|token|secret",line,re.I) else k+" [omitted]")
 r["aliases"][alias]={"returncode":p.returncode,"fields":fields,"stderr":p.stderr}
p=Path.home()/".ssh"
for v in p.iterdir():
 if v.is_file() or v.is_symlink():
  s=v.lstat();r["ssh_files"].append({"name":v.name,"mode":oct(stat.S_IMODE(s.st_mode)),"uid":s.st_uid,"bytes":s.st_size,"is_symlink":v.is_symlink()})
a=subprocess.run(["env","-u","LD_LIBRARY_PATH","ssh-add","-l"],capture_output=True,text=True);r["agent_public_fingerprints"]={"returncode":a.returncode,"stdout":a.stdout,"stderr":a.stderr}
print(json.dumps(r,sort_keys=True))
