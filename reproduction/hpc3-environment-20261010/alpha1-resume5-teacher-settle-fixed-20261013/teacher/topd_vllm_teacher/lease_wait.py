"""Repeat the same inherited lease checks until warmup utilization settles."""
import time
def wait_for_lease(preflight, expected=4, timeout=30., interval=1., clock=time.monotonic, sleep=time.sleep):
    deadline=clock()+timeout;retries=0
    while True:
        try:
            preflight.validate_inherited_gpu_leases(expected)
            return retries
        except RuntimeError as error:
            if 'exceeds the 15% idle thresholds' not in str(error) or clock()>=deadline:raise
            retries+=1
            print('TEACHER_PREWARM_IDLE_RECHECK retries='+str(retries),flush=True)
            sleep(interval)
