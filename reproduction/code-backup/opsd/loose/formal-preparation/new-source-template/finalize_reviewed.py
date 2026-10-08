"""Explicit post-review local regeneration. Never overwrite preview; never submit."""
import argparse,hashlib,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from verify_formal_prerequisites import loads
from render_staged import fresh_output,sha,replace_once

def finalize(preview,reviewed_sha,output):
 preview=Path(preview);mp=preview/'source-manifest.json'
 assert re.fullmatch('[0-9a-f]{64}',reviewed_sha) and sha(mp)==reviewed_sha, 'Exact independently reviewed preview manifest required'
 m=loads(mp.read_text());formal='formal_sources' in m
 names=m['formal_sources'] if formal else m['file_sources']
 for n,v in names.items():assert sha(preview/n)==v
 old=m['formal_directory_name'] if formal else m['directory_name']
 new=old+'-reviewed-'+reviewed_sha[:12]
 out=fresh_output(output)
 entries=('formal_entry.sh',) if formal else (('gpu_job_entry.sh',) if m['stage']=='pilot_only_no_future_ids_or_picked' else ('reload_job.sh',))
 for p in preview.iterdir():
  if not p.is_file() or p.name=='source-manifest.json':continue
  text=p.read_text()
  if p.name!='gpu_teacher_probe-factor-dtype.py':text=text.replace(old,new)
  if p.name in entries:
   text=replace_once(text,'echo "STAGED PREVIEW: independent review required" >&2\nexit 78\n','')
  (out/p.name).write_text(text)
 m['formal_directory_name' if formal else 'directory_name']=new
 if formal:
  m['formal_sources']={n:sha(out/n) for n in names};m['status']='review_enabled_final_requires_new_manifest_review'
 else:
  m['file_sources']={n:sha(out/n) for n in names}
  if 'runtime_sources' in m:m['runtime_sources']={n:sha(out/n) for n in m['runtime_sources']}
 m['preview_blocked']=False;m['reviewed_preview_manifest_sha256']=reviewed_sha
 (out/'source-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 return m
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--enable-reviewed',action='store_true',required=True);p.add_argument('--preview',required=True);p.add_argument('--reviewed-preview-sha',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 assert a.enable_reviewed
 finalize(a.preview,a.reviewed_preview_sha,a.output)
 print('NEW IMMUTABLE FINAL REGENERATED; FINAL MANIFEST/FILES REQUIRE INDEPENDENT REVIEW; NOT DEPLOYED',flush=True)
