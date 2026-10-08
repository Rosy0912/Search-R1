from pathlib import Path
import json
import os
import signal
import sys
import time

BASE=Path('/data/search-r1')
PIDFILE=BASE/'logs/retriever.pid'

def identity(pid):
    proc=Path('/proc')/str(int(pid))
    stat=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()
    args=proc.joinpath('cmdline').read_bytes().split(b'\0')
    return {'pid':int(pid),'start_ticks':stat[19],'uid':proc.stat().st_uid,
            'args':[a.decode() for a in args if a]}

def owned_retriever(record):
    try:
        now=identity(record['pid'])
        return now==record and str(BASE/'retriever_local.py') in now['args']
    except (OSError,KeyError,TypeError,ValueError):
        return False

if __name__=='__main__':
    if sys.argv[1]=='record':
        for attempt in range(40):
            record=identity(sys.argv[2])
            if owned_retriever(record):
                break
            time.sleep(0.05)
        assert owned_retriever(record), 'Retriever identity mismatch'
        tmp=PIDFILE.with_suffix('.pid.tmp')
        tmp.write_text(json.dumps(record))
        tmp.replace(PIDFILE)
    elif sys.argv[1]=='stop':
        try:
            record=json.loads(PIDFILE.read_text())
            if owned_retriever(record):
                fd=os.pidfd_open(record['pid'])
                try:
                    if owned_retriever(record):
                        signal.pidfd_send_signal(fd,signal.SIGTERM)
                finally:
                    os.close(fd)
        except (OSError,ValueError,TypeError):
            pass
