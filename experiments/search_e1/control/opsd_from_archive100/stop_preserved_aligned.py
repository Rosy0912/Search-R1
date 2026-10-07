from pathlib import Path
import os,json,time,signal
old=Path('/data/shaoshitong/search-e1-config-study-r1-aligned-8gpu-20261005');new=Path('/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007');run='b8b93546fb314915b3e72f355f22f291'
r=json.loads((old/'receipts/r1-round-1-grpo.json').read_text());assert r['run_id']==run
root=old/'preserved'/('replacement-state-'+run)
full=json.loads((root/'full-complete.json').read_text());assert full['run_id']==run and full['world_size']==8
assert sorted(x['rank'] for x in full['ranks'])==list(range(8))
assert all(Path(x['path']).stat().st_size==x['bytes'] for x in full['ranks'])
pid=r['pid'];assert os.getpgid(pid)==pid
assert (Path('/proc')/str(pid)).stat().st_uid==os.getuid()
assert ('E1_RUN_ID='+run).encode() in (Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0')
owned=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if os.getpgid(int(p.name))!=pid:continue
  assert p.stat().st_uid==os.getuid()
  owned.append(int(p.name))
 except (FileNotFoundError,ProcessLookupError):pass
assert pid in owned
proof=dict(human_authorization=json.loads((new/'receipts/authorization.json').read_text()),run_id=run,private_group=pid,owned_pids=owned,preserved_state_step=full['global_step'],full_state=str(root/'full-complete.json'),time=time.time(),reason='Latest human replacement; stop only current owned aligned8 GRPO; archive4 unchanged')
for b in (old,new):(b/'receipts/aligned-user-replacement-stop.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2))
os.killpg(pid,signal.SIGTERM)
print(json.dumps(proof,ensure_ascii=False),flush=True)
