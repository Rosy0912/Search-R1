import pathlib,json,datetime,os
root=pathlib.Path('/mnt/data/shaoshitong/search-g1-mixed-first-independent-backup-20261011/job3364-15bb15365346-step5-once-v1')
o=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),namespace=str(root))
for n in ('RECEIPT.json','PROGRESS.json','RESULT.json'):
 p=root/n
 if p.exists():o[n]=json.loads(p.read_text())
r=o.get('RECEIPT.json',{});p=pathlib.Path('/proc')/str(r.get('pid',0))/'stat'
if p.exists():
 fields=p.read_text().rsplit(')',1)[1].split();o['original_process_alive']=int(fields[19])==r['start_ticks'];o['process_state']=fields[0]
else:o['original_process_alive']=False
p=root/'backup.log'
if p.exists():
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-3000));o['tail']=f.read().decode(errors='replace')
staging=root/'checkpoint.partial';o['actual_staging_file_bytes']=sum(p.stat().st_size for p in staging.rglob('*') if p.is_file()) if staging.exists() else None
final=root/'checkpoint';o['final_directory_exists']=final.is_dir();o['actual_final_file_count']=sum(1 for p in final.rglob('*') if p.is_file()) if final.exists() else 0;o['actual_final_file_bytes']=sum(p.stat().st_size for p in final.rglob('*') if p.is_file()) if final.exists() else 0
print(json.dumps(o))
