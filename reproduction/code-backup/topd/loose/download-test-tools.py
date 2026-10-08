import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

stage = Path('/Users/shaoshitong/Documents/remote_codex_control/.transfer-staging/topd-test-tools-20261003')
stage.mkdir(exist_ok=True)
opener = urllib.request.build_opener(urllib.request.ProxyHandler(
    {'http':'http://127.0.0.1:7892','https':'http://127.0.0.1:7892'}))
records = []
for name, version in [('pytest','8.3.5'),('pluggy','1.5.0'),('iniconfig','2.0.0'),
                       ('exceptiongroup','1.2.2'),('tomli','2.2.1'),('packaging','24.2')]:
    with opener.open(f'https://pypi.org/pypi/{name}/{version}/json',timeout=60) as response:
        metadata = json.load(response)
    wheel = next(item for item in metadata['urls'] if item['filename'].endswith('py3-none-any.whl'))
    path = stage/wheel['filename']
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != wheel['digests']['sha256']:
        with opener.open(wheel['url'],timeout=60) as response:
            data = response.read()
        assert hashlib.sha256(data).hexdigest() == wheel['digests']['sha256']
        path.write_bytes(data)
    subprocess.run([sys.executable,'work/transfer_file.py',str(path),
                    '/dataset/shaoshitong/search-r1-rl/topd/cache/test-wheelhouse/'+path.name],check=True)
    records.append({'package':name,'version':version,'file':path.name,'sha256':wheel['digests']['sha256'],
                    'url':wheel['url'],'target_verified':True})
    (stage/'manifest.json').write_text(json.dumps(records,indent=2))
    print('PINNED_TEST_WHEEL_SOURCE_AND_TARGET_VERIFIED',name,version,flush=True)
Path('outputs/test-tools-manifest.json').write_text(json.dumps(records,indent=2))
