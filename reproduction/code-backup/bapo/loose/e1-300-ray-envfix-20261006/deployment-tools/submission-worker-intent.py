import sys,pathlib,json,datetime
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006')
phase=sys.argv[1];assert phase in ('hardware','formal300')
assert root.is_dir()and not root.is_symlink()
assert not any((root/'runs').glob('bapo-instruct300-'+phase+'-*')),'Existing phase run; reconcile, do not submit'
p=root/'logs'/('submission-intent-'+phase+'.json')
with p.open('x')as f:json.dump({'phase':phase,'source_commit':(root/'source-expected.txt').read_text().strip(),'time':datetime.datetime.now(datetime.timezone.utc).isoformat()},f)
print('DURABLE_WORKER_SUBMISSION_INTENT_RECORDED',phase,flush=True)
