"""Exact isolated sourceV2 identity shared by preparation and actual launch."""
import pathlib
SOURCE=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/tools/search-r1-32b-grpo100-source-v2')
SOURCE_SHA='9040e2ee665e346f2c4c37d3319e65c32b67612668c4f7088c5f6c68c72106d0'
def check(c):
 ref=c['source_manifest']
 if pathlib.Path(c['candidate'])!=SOURCE or pathlib.Path(ref['path'])!=SOURCE/'r1-files.json' or ref['sha256']!=SOURCE_SHA:raise ValueError('Exact isolated streamed sourceV2 binding required')
 return SOURCE_SHA
