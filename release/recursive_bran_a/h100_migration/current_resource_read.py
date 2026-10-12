"""Read current run-scoped observer and disk; no GPU allocation or state mutation."""
import json
import pathlib
import shutil
import time

base = pathlib.Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012')
observer = base / 'observers/run-1c806d-step3389-0'
binding = json.loads((observer / 'binding.json').read_text())
proc = pathlib.Path('/proc/3740165')
stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
assert proc.stat().st_uid == 0 and int(stat[19]) == 658259848
memory = json.loads((observer / 'control/memory-live.json').read_text())
assert memory['observer_pid'] == 3740165
assert memory['run_id'] == '1c806d15752d44a7ac99687f0737d7e3'
assert memory['job'] == '3389'
now = time.time()
print(json.dumps(dict(time=now, binding=binding, observer_process=dict(
    pid=3740165, uid=0, starttick=int(stat[19]), state=stat[0]),
    memory=memory, age_seconds=now-memory['time'], disk_free_bytes=shutil.disk_usage(base).free,
    scope='Fresh run observer; sampled peaks since observer start, not hardware maximum or first-outer full coverage')))
