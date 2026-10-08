import pathlib,json,hashlib,time
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');run=root/'runs/tree-fixed-two-3088';tools=root/'candidates/tree-fixed-two-296217a-v2';cand=root/'candidates/tree-native-numeric-296217a';files={}
paths=[run/'launch-manifest.json',run/'resolved.yaml',tools/'manifest.json',tools/'supervisor.py',tools/'gpu-kl-proof.py',tools/'sitecustomize.py',root/'tools/start_tree_fixed_two_296217a_v2.sh',cand/'repo/verl/trainer/ppo/core_algos.py']+[run/'gpu-kl-proof'/f'rank-{r}.json' for r in range(8)]
for p in paths:
 data=p.read_bytes();files[str(p)]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'text':data.decode()}
print(json.dumps({'epoch':time.time(),'files':files}))
