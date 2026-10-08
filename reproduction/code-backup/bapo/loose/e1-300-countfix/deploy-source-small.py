import pathlib,hashlib,subprocess,json
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-countfix-20261004');assert r.is_dir()and not r.is_symlink()
s='f0e37bf68d61ec926ac225f5e1ee1f606cb09bf9';p=r/'source.bundle'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='a7ba350d59781d02c7fa05172114a6861958febdf0dbb14d53f3eae606c98187'
assert not (r/'repo').exists()
subprocess.run(['git','clone','--no-checkout',str(p),str(r/'repo')],check=True)
subprocess.run(['git','-C',str(r/'repo'),'checkout','--detach',s],check=True)
assert subprocess.check_output(['git','-C',str(r/'repo'),'rev-parse','HEAD'],text=True).strip()==s
assert not subprocess.check_output(['git','-C',str(r/'repo'),'status','--porcelain'],text=True)
(r/'source-expected.txt').write_text(s+'\n')
print(json.dumps({'status':'NEW_STATEFIX_SOURCE_DEPLOYED_NOT_GPU_ACCEPTED','root':str(r),'source_commit':s,'bundle_sha256':'a7ba350d59781d02c7fa05172114a6861958febdf0dbb14d53f3eae606c98187'}))
