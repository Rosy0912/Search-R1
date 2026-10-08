import pathlib,json,os,socket,subprocess
roots=['/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/assets/qwen2.5-3b-instruct-aa8e725','/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-20261004/assets/qwen2.5-3b-instruct-aa8e725']
records=[]
for root in roots:
 p=pathlib.Path(root);r={'path':root,'exists':p.exists()}
 if p.exists():
  st=p.stat();r.update(device=st.st_dev,inode=st.st_ino,files={f.name:{'bytes':f.stat().st_size,'device':f.stat().st_dev,'inode':f.stat().st_ino} for f in p.iterdir() if f.is_file()})
 q=subprocess.run(['findmnt','-T',root,'-n','-o','SOURCE,FSTYPE,TARGET'],capture_output=True,text=True);r['mount']=q.stdout.strip();r['mount_error']=q.stderr.strip();records.append(r)
print(json.dumps({'node':socket.gethostname(),'roots':records},sort_keys=True))
