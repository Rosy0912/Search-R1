"""Unknown read state never becomes release evidence; no process signals."""
import time


def await_release(read,record,sleep,clock=time.monotonic,opaque_grace_seconds=55):
    start=clock()
    while True:
        try:
            proof=read()
        except Exception as error:
            # Keep controller/locks alive. No unreadable proc/driver becomes
            # absent/foreign, even after a deadline.
            try:record(dict(status='UNCERTAIN_READ_RETRY_LOCKS_HELD',error=repr(error),released=False))
            except Exception:pass # Even disk write failure cannot grant release.
            sleep(5);continue
        try:record(proof)
        except Exception:
            sleep(5);continue # Cannot preserve release evidence: fail closed.
        if proof['released']:return proof
        if (proof.get('known_identities_absent_or_reused') is True and proof.get('own_cuda_clear') is True and
                proof.get('only_independent_opaque_blockers') is True and clock()-start>=opaque_grace_seconds):
            raise RuntimeError('HOLD: independent opaque audit blockers; known owned/CUDA release verified')
        sleep(5)
