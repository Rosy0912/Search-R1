"""One-shot recovery for this upload only; never stops a live writer or training."""
import os,sys,time,pathlib,json,fcntl,subprocess
ROOT=pathlib.Path(__file__).resolve().parent
superlock=(ROOT/'recovery-supervisor.lock').open('w')
fcntl.flock(superlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
token=sys.stdin.readline().strip();assert token
assert ROOT.name=='h100-cleanup-20261012-gigpo'
assert json.loads((ROOT/'plan.json').read_text())['repo']=='Sutoonq/TMP_MODEL'
def write(value):
 p=ROOT/'recovery-supervisor.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(epoch=time.time(),pid=os.getpid(),**value)));t.replace(p)
def writer_busy():
 p=(ROOT/'uploader.lock').open('a')
 try:fcntl.flock(p,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:p.close();return True
 fcntl.flock(p,fcntl.LOCK_UN);p.close();return False
resumes=0
write({'status':'WAITING_CURRENT_WRITER_OR_VERIFIED_COMPLETION','resumes':resumes})
while not (ROOT/'upload-complete.json').exists():
 if writer_busy():time.sleep(30);continue
 if resumes>=3:
  write({'status':'NEEDS_ATTENTION_BOUNDED_RECOVERY_EXHAUSTED','resumes':resumes});raise SystemExit(1)
 resumes+=1
 out=(ROOT/('recovery-upload-'+str(resumes)+'.stdout')).open('ab',buffering=0)
 err=(ROOT/('recovery-upload-'+str(resumes)+'.stderr')).open('ab',buffering=0)
 env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None)
 proc=subprocess.Popen([sys.executable,str(ROOT/'upload.py')],cwd=ROOT,stdin=subprocess.PIPE,stdout=out,stderr=err,env=env,start_new_session=True)
 proc.stdin.write((token+'\n').encode());proc.stdin.close()
 write({'status':'DETACHED_RECEIPT_VERIFIED_RECOVERY_RUNNING','resumes':resumes,'uploader_pid':proc.pid})
 code=proc.wait();out.close();err.close()
 write({'status':'RECOVERY_EXITED_CHECKING_COMPLETION','resumes':resumes,'uploader_pid':proc.pid,'exit_code':code})
 time.sleep(5)
complete=json.loads((ROOT/'upload-complete.json').read_text());assert complete['status']=='ALL_SELECTED_HF_FILES_SHA_VERIFIED';assert complete['files']==46
logs=ROOT.with_name(ROOT.name+'-logs');lc=json.loads((logs/'upload-complete.json').read_text());assert lc['status']=='ALL_SELECTED_HF_FILES_SHA_VERIFIED'
cleanup=json.loads((ROOT/'cleanup-complete.json').read_text());assert cleanup['status']=='BEST_HF_INFERENCE_SHA_VERIFIED_SOURCE_FRESH_CHECKED_OLD_OUTPUTS_CLEANED_TRAINING_ADVANCED'
write({'status':'ALL_CHECKPOINTS_AND_LOGS_SHA_VERIFIED_CLEANUP_TRAINING_ADVANCED','resumes':resumes,'checkpoint_complete':complete,'logs_complete':lc,'cleanup_complete':cleanup})
