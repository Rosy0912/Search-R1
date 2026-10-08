"""Read-only process samples confer no registration or signalling authority."""
import errno,pathlib,time

def observe(tools,pid,previous=None,proc='/proc'):
 root=pathlib.Path(proc)/str(pid)
 def gone(error=None):
  try:root.stat()
  except OSError as e:
   if e.errno in (errno.ENOENT,errno.ESRCH):return {'pid':pid,'scope':'OBSERVED_GONE_NON_AUTHORITY','observed_epoch':time.time(),'error':repr(error) if error else None,'fresh_proc_absent':True}
   raise
  raise RuntimeError('Process observation unknown while actual proc remains') from error
 try:
  ident=tools.process_identity(pid)
  if ident.get('identity_unavailable') or 'starttime_ticks' not in ident:return gone()
  uid=int(next(line for line in (root/'status').read_text().splitlines() if line.startswith('Uid:')).split()[1]);ident['uid']=uid
  sample=tools.process_sample(pid)
  if sample.get('exited'):return gone()
  after=tools.process_identity(pid)
  if after.get('identity_unavailable') or 'starttime_ticks' not in after:return gone()
  after_uid=int(next(line for line in (root/'status').read_text().splitlines() if line.startswith('Uid:')).split()[1]);after['uid']=after_uid
  if after['starttime_ticks']!=ident['starttime_ticks'] or after_uid!=uid or (previous and 'starttime_ticks' in previous and (previous['starttime_ticks']!=ident['starttime_ticks'] or previous.get('uid',uid)!=uid)):raise RuntimeError('Live observed PID reuse/UID drift')
  if after.get('cgroup')!=ident.get('cgroup'):
   return {'pid':pid,'scope':'OBSERVED_IDENTITY_TRANSITION_NON_AUTHORITY','observed_epoch':time.time(),'before_identity':ident,'after_identity':after,'inconclusive':True,'trusted_rss_sample':None}
  return {'pid':pid,'scope':'OBSERVED_PROC_RSS_CPU_NON_AUTHORITY','identity':ident,'after_identity':after,'sample':sample,'observed_epoch':time.time()}
 except OSError as e:
  if e.errno in (errno.ENOENT,errno.ESRCH):return gone(e)
  raise
