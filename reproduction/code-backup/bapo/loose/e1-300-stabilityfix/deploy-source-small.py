import pathlib,hashlib,subprocess,json
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-20261005');assert r.is_dir()and not r.is_symlink()
s='39cdcdd1404213b897bb0de99cc52d72707682f0';p=r/'source-verified.bundle'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='40e2165b862e9b5b7432098c860b6596e78b9e08f6cec5450034eabb5eea0eaa'
assert not (r/'repo').exists()
subprocess.run(['git','clone','--no-checkout',str(p),str(r/'repo')],check=True)
subprocess.run(['git','-C',str(r/'repo'),'checkout','--detach',s],check=True)
assert subprocess.check_output(['git','-C',str(r/'repo'),'rev-parse','HEAD'],text=True).strip()==s
assert not subprocess.check_output(['git','-C',str(r/'repo'),'status','--porcelain'],text=True)
(r/'source-expected.txt').write_text(s+'\n')
print(json.dumps({'status':'NEW_STATEFIX_SOURCE_DEPLOYED_NOT_GPU_ACCEPTED','root':str(r),'source_commit':s,'bundle_sha256':'40e2165b862e9b5b7432098c860b6596e78b9e08f6cec5450034eabb5eea0eaa'}))
