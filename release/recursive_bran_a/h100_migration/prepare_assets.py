import json,os,secrets,pathlib
b=pathlib.Path(os.environ['RECURSIVE_BRAN_BASE'])/'retriever';r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009')
m=json.loads((r/'inputs/baseline-aligned-retrieval.json').read_text());assets=b/'assets';assets.mkdir(exist_ok=True)
for relative,key in [('index/e5_Flat.index','index'),('corpus/wiki-18.jsonl','corpus'),('encoder','encoder/config.json')]:
 target=pathlib.Path(m['assets'][key]['path']);target=target.parent if relative=='encoder' else target
 assert target.exists(),str(target)
 link=assets/relative;link.parent.mkdir(exist_ok=True)
 if not link.exists():link.symlink_to(target)
(assets/'manifest.json').write_text(json.dumps({'reused_h100_assets':m['assets']},indent=2))
token=b/'receipts/private-probe-token';token.write_text(secrets.token_hex(32));token.chmod(0o600)
