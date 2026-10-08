import pathlib,hashlib,os,json
old=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-precision-r3-20261004/source-verified.bundle');new=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-precision-r4-20261004/source-verified.bundle');expected='fbdb6943edd3aa05997c76e034431d84900b22d100af1e9d6c344e85718e0def'
assert old.is_file()and not old.is_symlink()and hashlib.sha256(old.read_bytes()).hexdigest()==expected
if not new.exists():os.link(old,new)
assert new.is_file()and not new.is_symlink()and hashlib.sha256(new.read_bytes()).hexdigest()==expected
print(json.dumps({'status':'VERIFIED_SOURCE_BUNDLE_REUSED_OLD_PARTIAL_UPLOAD_PRESERVED','path':str(new),'sha256':expected,'bytes':new.stat().st_size}))
