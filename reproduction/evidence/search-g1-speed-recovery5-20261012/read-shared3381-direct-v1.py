from pathlib import Path
import sys,shlex,datetime,json,subprocess
sys.path.insert(0,"work")
import search_g1_restore5_transport as t
original=subprocess.run
def direct(args,*a,**kw):
    if isinstance(args,list) and "mgmt-host" in args:
        args=list(args); option="ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p"
        if option in args:
            i=args.index(option);del args[i-1:i+1]
    return original(args,*a,**kw)
t.subprocess.run=direct
stamp=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
script=sys.stdin.read()
x=t.remote("/usr/bin/python3 -c "+shlex.quote(script),f"outputs/gigpo-e1-instruct200/SEARCH-G1-SHARED3381-DIRECT-{stamp}.json",emit=False)
print(x)
