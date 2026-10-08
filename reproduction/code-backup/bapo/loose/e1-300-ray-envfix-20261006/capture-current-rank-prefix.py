import pathlib,json,hashlib,base64,time
run=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006/runs/bapo-instruct300-formal300-3292')
source='a2568d0df306ead6e059760e28b5bd53ee3f6876'
assert (run/'source-commit.txt').read_text().strip()==source
p=run/'train.log';records=[];h=hashlib.sha256();offset=0;end=0;sha=None;nrecords=0;laststep=None
with p.open('rb') as f:
 for line in f:
  h.update(line)
  for marker in (b'ACTOR_RANK_UPDATE_JSON=',b'BAPO_TRAIN_METRICS_JSON='):
   if marker in line:
    records.append(dict(offset=offset,bytes=len(line),sha256=hashlib.sha256(line).hexdigest(),line_base64=base64.b64encode(line).decode()))
    if marker==b'BAPO_TRAIN_METRICS_JSON=':
     d=json.JSONDecoder().raw_decode(line.split(marker,1)[1].decode().lstrip())[0]
     end=offset+len(line);sha=h.hexdigest();nrecords=len(records);laststep=d['step']
  offset+=len(line)
assert end and laststep>=26
print(json.dumps(dict(job_id=3292,source_commit=source,remote_path=str(p),captured_time=time.time(),snapshot_cutoff_bytes=end,snapshot_cutoff_sha256=sha,last_complete_outer=laststep,records=records[:nrecords],scope='EXACT_MARKER_LINES_WITH_ORIGINAL_OFFSETS_NOT_FULL_LOG')))
