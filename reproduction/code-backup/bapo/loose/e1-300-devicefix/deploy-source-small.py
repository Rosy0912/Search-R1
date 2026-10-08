import pathlib,hashlib,subprocess,json
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-devicefix-20261004');assert r.is_dir()and not r.is_symlink()
s='733097feeae371b4efd8fd48940c46ff36da839b';p=r/'source.bundle'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='fbdb6943edd3aa05997c76e034431d84900b22d100af1e9d6c344e85718e0def'
assert not (r/'repo').exists()
subprocess.run(['git','clone','--no-checkout',str(p),str(r/'repo')],check=True)
subprocess.run(['git','-C',str(r/'repo'),'checkout','--detach',s],check=True)
assert subprocess.check_output(['git','-C',str(r/'repo'),'rev-parse','HEAD'],text=True).strip()==s
assert not subprocess.check_output(['git','-C',str(r/'repo'),'status','--porcelain'],text=True)
(r/'source-expected.txt').write_text(s+'\n')
print(json.dumps({'status':'NEW_STATEFIX_SOURCE_DEPLOYED_NOT_GPU_ACCEPTED','root':str(r),'source_commit':s,'bundle_sha256':'fbdb6943edd3aa05997c76e034431d84900b22d100af1e9d6c344e85718e0def'}))
