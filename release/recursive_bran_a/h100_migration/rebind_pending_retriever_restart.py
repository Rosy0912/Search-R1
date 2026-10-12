"""Refresh a not-yet-consumed launch after its allocation killed CPU services."""
import json
from pathlib import Path
import time

b=Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012')
assert not (b/'receipts/train-start.json').exists()
assert not Path('/proc/3504331').exists()
ready=b/'control/launch-ready.sh'
assert ready.exists()
source=b/'source-71d4e69'
script=(source/'release/recursive_bran_a/h100_migration/run.sh').read_text()
script=script.replace('source="$(cd "$(dirname "$0")/../../.." && pwd)"','source="'+str(source)+'"')
configure='"$python" -B release/recursive_bran_a/h100_migration/prepare_train.py > "$base/logs/configure.log" 2>&1\n'
assert configure in script
script=script.replace(configure,'# Preserve the approved run and immutable scientific configuration.\n')
preserved=b/'preserved'/('expired-3387-service-receipts-'+str(int(time.time())))
preserved.mkdir(parents=True,exist_ok=False)
(b/'retriever/receipts').rename(preserved/'retriever-receipts')
(b/'retriever/receipts').mkdir()
ready.rename(preserved/'previous-launch-ready.sh')
ready.write_text(script)
print(json.dumps({'prepared_pending_launch':True,'config_unchanged':True,'run':json.loads((b/'receipts/configuration.json').read_text())['run_id'],'old_service_receipts_preserved':str(preserved),'service_identity_changed_requires_new_gate':True}),flush=True)
