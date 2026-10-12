import pathlib,json,subprocess,time
base=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006/runs')
roots=[base/n for n in ['gigpo-e1-fresh200-live-job3340-1791441919057393097','gigpo-e1-fresh200-live-job3344-1791482568900917579','gigpo-e1-fresh200-live-job3347-1791494104474180576']];records=[]
for root in roots:
 for p in list(root.glob('actual-rollout-attempt*.pt'))+list(root.glob('audit/partial-updates/attempt_*/driver.pt')):
  assert p.is_file() and not p.is_symlink();s=p.stat();records.append({'path':str(p),'bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'mtime_ns':s.st_mtime_ns,'inode':s.st_ino})
print(json.dumps({'epoch':time.time(),'roots':[str(p) for p in roots],'records':records,'bytes':sum(r['bytes'] for r in records),'allocated_bytes':sum(r['allocated_bytes'] for r in records)}))
