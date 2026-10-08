import pathlib,hashlib,subprocess,json
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-statefix-20261004');assert r.is_dir()and not r.is_symlink()
s='2175549691d6856e2b393b2834a32febe41ee56e';p=r/'source.bundle'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='02d8e7d02422d3fa4590b58857bded4a175528cf93a475747fe509cca2667cc0'
assert not (r/'repo').exists()
subprocess.run(['git','clone','--no-checkout',str(p),str(r/'repo')],check=True)
subprocess.run(['git','-C',str(r/'repo'),'checkout','--detach',s],check=True)
assert subprocess.check_output(['git','-C',str(r/'repo'),'rev-parse','HEAD'],text=True).strip()==s
assert not subprocess.check_output(['git','-C',str(r/'repo'),'status','--porcelain'],text=True)
(r/'source-expected.txt').write_text(s+'\n')
print(json.dumps({'status':'NEW_STATEFIX_SOURCE_DEPLOYED_NOT_GPU_ACCEPTED','root':str(r),'source_commit':s,'bundle_sha256':'02d8e7d02422d3fa4590b58857bded4a175528cf93a475747fe509cca2667cc0'}))
