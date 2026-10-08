import pathlib,subprocess,json
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006');old=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-titlefix-r2-20261006/repo');source='a2568d0df306ead6e059760e28b5bd53ee3f6876'
assert root.resolve()==root and root.is_dir() and not(root/'repo').exists() and not(root/'source-expected.txt').exists()
assert subprocess.check_output(['git','-C',str(old),'rev-parse','HEAD'],text=True).strip()=='3f85dabc7aa05bc45ec7af015d65d9fa4a4c17e4'
assert not subprocess.check_output(['git','-C',str(old),'status','--porcelain'],text=True).strip()
subprocess.run(['git','clone','--no-hardlinks',str(old),str(root/'repo')],check=True)
subprocess.run(['git','-C',str(root/'repo'),'fetch',str(root/'logs/source-a256.bundle'),'refs/heads/codex/bapo-instruct300-ray-envfix'],check=True)
subprocess.run(['git','-C',str(root/'repo'),'checkout','--detach',source],check=True)
assert subprocess.check_output(['git','-C',str(root/'repo'),'rev-parse','HEAD'],text=True).strip()==source
assert not subprocess.check_output(['git','-C',str(root/'repo'),'status','--porcelain'],text=True).strip()
with(root/'source-expected.txt').open('x') as f:f.write(source+'\n')
print(json.dumps(dict(status='NEW_ISOLATED_SOURCE_DEPLOYED_NOT_GPU_OR_TRAINING',source=source,root=str(root))))
