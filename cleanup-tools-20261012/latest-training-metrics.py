import pathlib,json,time
p=pathlib.Path('/mnt/data/shaoshitong/search-g1-speed-resume5-0553ade-20261012/logs/trainer.log')
with p.open('rb') as f:f.seek(max(0,p.stat().st_size-4*1024**2));data=f.read().decode(errors='replace')
rows=[s for s in data.splitlines() if 'step:' in s and 'train/' in s]
print(json.dumps({'epoch':time.time(),'metric_lines':rows[-2:]}))
