import hashlib,json,pathlib,subprocess,urllib.request
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');dest=root/'cpu-blas-experiment';dest.mkdir(exist_ok=True)
meta=subprocess.check_output(['apt-cache','show','libopenblas0-pthread'],text=True).split('\n\n')[0];m=dict(line.split(': ',1) for line in meta.splitlines() if ': ' in line)
assert m['Version']=='0.3.20+ds-1' and m['Architecture']=='amd64'
p=dest/pathlib.Path(m['Filename']).name;opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://10.33.0.16:3128'}));data=opener.open('https://mirrors.aliyun.com/ubuntu/'+m['Filename'],timeout=120).read()
assert len(data)==int(m['Size']) and hashlib.sha256(data).hexdigest()==m['SHA256'];p.write_bytes(data)
subprocess.run(['dpkg-deb','-x',str(p),str(dest)],check=True)
assert (dest/'usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so.0').is_file()
record={'package_metadata':m,'package_sha256_verified':True,'scope':'CPU microbenchmark only; ownroot extraction, no system or production environment change','installed_runtime':subprocess.check_output(['dpkg-query','-W','python3.10'],text=True)}
(root/'logs/cpu-blas-experiment-package.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
