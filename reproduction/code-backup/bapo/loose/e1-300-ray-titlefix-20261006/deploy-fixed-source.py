import base64,hashlib,json,subprocess
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-titlefix-20261006')
old=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-freeze-scalability-20261006/repo')
source='484c0b30fa73fa5354bdb3452f007134ec5db13f'
assert r.is_dir() and not r.is_symlink() and not (r/'repo').exists() and not (r/'source-expected.txt').exists()
raw=base64.b64decode('IyB2MiBnaXQgYnVuZGxlCi1iNzExOTc5MWI4Njk1MTcyN2NlYTM2YTE4ODFlOGE1MjVjOTMyNDk0IEJpbmQgc2NhbGFibGUgcHJvY2VzcyBmcmVlemUgYW5kIHN0cmljdCBwYXJlbnQgZXBvY2ggam91cm5hbCB2ZXJpZmljYXRpb24KNDg0YzBiMzBmYTczZmE1MzU0YmRiMzQ1MmYwMDcxMzRlYzVkYjEzZiBIRUFECgpQQUNLAAAAAgAAAAWQEnicnY9RTsUgEEX/u4r5N3lhoBRIjDEuwR0MMEa0hRc67bO7ly8X4N/Nyc25udKZQSNpNNoxRm2Cnz+cQVriooh5jk4PgD5rO92pcxWIDjG4gNEvwaLTLjGZhdB7ZE9W2xSMnsM80SGfrQPVVq+tHTs8/8VX/qHtvvKt1JPWkl8Ah3HMW63gSXmlptS2rYjwvwXTW6kZOp+FH5zhnS4Q2r9BiqwMJY8vRa5hz7BLL0lGN7WT+wVf7eiV1ukXN9Vb/vACyaRmNDOKR1JzyPCtB8etBnlHGzZ4nOti62LboMQksuO/uHDmt0tpuwSnmNy1cuNv3D/l1WQzpisA1C0N3f4B/3CzGulvU4EOmLp5qvyBcU5Gx054nNvFuItxgrmI0O6wM9f4JNb8Sopv+hufVVLF/7p2onc+AMgMDZzwAtjUBRj16n7HYcap4PYoDtcJsLLCeJxbw7aGbcNRJhFFwbtPwyPr2o4lJ6wRqZSpesbl3jP5JlMwANumDXT1CgpLSNe9REPQMm2bBp04lWEVnK2jeJwNyrsNwjAQAFCFkgUYIe3Zvo9dUERRpHRICYgysn2XAoZgJcQ62SM1kV759q3Zt+b7aj6OzaEXp7CKOFCpkasBmM9kDooeVAgTQDFJ5Apl8B4tkSARtedpmG+PqR+WZ3fvx2Ueu2tLwZS911oNcyYU0ggrY0wcQ2TIYcWgou5S2LDycZiEManPQjnC77S//9j3L1vsDIjCBn1uQvsAsTncLjs8/6NPUA==')
assert hashlib.sha256(raw).hexdigest()=='6d57c1ed074e36208b63d4e25129483de20f35b15b0669892edf8dfd8c73749e'
with (r/'logs/source-484c.bundle').open('xb') as f:f.write(raw)
subprocess.run(['git','clone','--no-hardlinks','--no-checkout',str(old),str(r/'repo')],check=True)
subprocess.run(['git','-C',str(r/'repo'),'fetch',str(r/'logs/source-484c.bundle'),'HEAD'],check=True)
subprocess.run(['git','-C',str(r/'repo'),'checkout','--detach',source],check=True)
assert subprocess.check_output(['git','-C',str(r/'repo'),'rev-parse','HEAD'],text=True).strip()==source
assert not subprocess.check_output(['git','-C',str(r/'repo'),'status','--porcelain'],text=True).strip()
with (r/'source-expected.txt').open('x') as f:f.write(source+'\n')
print(json.dumps({'source':source,'root':str(r),'source_clean':True,'tools_deployed':False,'training_submitted':False}))
