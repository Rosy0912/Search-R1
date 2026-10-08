import json
from download_assets import BASE, valid
assets=json.loads((BASE/'assets.json').read_text())
failed=[a['path'] for a in assets if not valid(BASE/'assets'/a['path'],a)]
print(json.dumps({'asset_count':len(assets),'content_hashes_verified':len(assets)-len(failed),'failed':failed},indent=2))
assert not failed, 'Official content verification failed'
