import hashlib,json,pathlib,subprocess,urllib.request
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');dest=root/'python-dev';dest.mkdir(exist_ok=True)
meta=subprocess.check_output(['apt-cache','show','libpython3.10-dev'],text=True).split('\n\n')[0];m=dict(line.split(': ',1) for line in meta.splitlines() if ': ' in line)
assert m['Version'].startswith('3.10.12-') and m['Architecture']=='amd64'
p=dest/pathlib.Path(m['Filename']).name;opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://10.33.0.16:3128'}));data=opener.open('https://mirrors.aliyun.com/ubuntu/'+m['Filename'],timeout=120).read()
assert len(data)==int(m['Size']) and hashlib.sha256(data).hexdigest()==m['SHA256'];p.write_bytes(data)
subprocess.run(['dpkg-deb','-x',str(p),str(dest)],check=True)
assert (dest/'usr/include/python3.10/Python.h').is_file() and (dest/'usr/include/x86_64-linux-gnu/python3.10/pyconfig.h').is_file()
record={'package_metadata':m,'package_sha256_verified':True,'scope':'extract to ownroot only; no dpkg install/system change; only headers used','installed_runtime':subprocess.check_output(['dpkg-query','-W','python3.10'],text=True)}
(root/'logs/python-headers-acceptance.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
