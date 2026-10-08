import urllib.request,json,hashlib,pathlib,subprocess
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');m=json.loads((root/'tools/click-compatibility.json').read_text());p=root/'wheelhouse'/m['filename']
opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://10.33.0.16:3128'}))
data=opener.open(m['url'],timeout=60).read();assert hashlib.sha256(data).hexdigest()==m['sha256'];p.write_bytes(data)
subprocess.run([str(root/'tools/uv'),'pip','install','--python',str(root/'env/bin/python'),'--no-deps',str(p)],check=True)
subprocess.run([str(root/'env/bin/ray'),'--version'],check=True)
