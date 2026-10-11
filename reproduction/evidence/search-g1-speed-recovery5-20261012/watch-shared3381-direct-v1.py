import subprocess,runpy
original_run=subprocess.run
def direct_run(args,*a,**kw):
    if isinstance(args,list) and "mgmt-host" in args:
        args=list(args)
        option="ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p"
        if option in args:
            i=args.index(option)
            del args[i-1:i+1]
    return original_run(args,*a,**kw)
subprocess.run=direct_run
runpy.run_path("work/watch_search_g1_reduced_gate3381_personal_five_v2.py",run_name="__main__")
