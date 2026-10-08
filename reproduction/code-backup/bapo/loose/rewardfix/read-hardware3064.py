import pathlib,json,hashlib,subprocess
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-b097');run=r/'runs/bapo-hardware-3064'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
manifest=json.loads((r/'tools/hardware-launch-hashes-b09736d.json').read_text())
files=[p for p in run.glob('*') if p.is_file()]+[r/'tools'/n for n in manifest]+[r/'tools/hardware-launch-hashes-b09736d.json']
files += [r/p for p in ['logs/environment-acceptance.json','assets-root/assets.json','logs/derived-assets.sha256','logs/common-split-manifest.json','logs/shared-inputs.json','logs/isolation-setup.json','logs/cpu-full-b097.txt','logs/cpu-gate-b097.json','logs/stage64-b097-proof.json','logs/hardware-launch-approval-b097.json','logs/bapo-rewardfix-independent-launch-review.md','logs/hardware3064-terminal-slurm.log','logs/hardware3064-archive-manifest.json','logs/archive-partial-own-process-handling.json','logs/bapo-rewardfix-hardware3064-raw-measurements.tar.gz'] if (r/p).is_file()]
proof=json.loads((r/'logs/stage64-b097-proof.json').read_text().split('BAPO_STAGE64_DERIVATION_PROOF_JSON=')[-1]);p=pathlib.Path(proof['stage_file']);files += [p,p.with_name('manifest.json')]
content={str(p.relative_to(r)):p.read_text() for p in files if p.suffix in ['.json','.jsonl','.log','.csv','.txt'] or p.name in ['hardware.log','source-commit.txt']}
print(json.dumps({'source':subprocess.check_output(['git','-C',str(r/'repo'),'rev-parse','HEAD'],text=True).strip(),'dirty':subprocess.check_output(['git','-C',str(r/'repo'),'status','--porcelain'],text=True),'files':[{'path':str(p),'sha256':sha(p),'resolved':str(p.resolve()),'size':p.stat().st_size} for p in files],'manifest':manifest,'content':content}))
