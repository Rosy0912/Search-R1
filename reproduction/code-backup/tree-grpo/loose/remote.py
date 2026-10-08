import argparse, shlex, subprocess, sys

def route(command, node="h100-9"):
    control = "/dev/shm/shaoshitong-tree-grpo-control-20261003"
    setup = "mkdir -p " + control + "; chmod 700 " + control + "; "
    opts = "-T -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ControlMaster=auto -o ControlPersist=600 -o ControlPath=" + control + "/%h-%p"
    worker = "env -u LD_LIBRARY_PATH ssh " + opts + " " + node + " " + shlex.quote(command)
    if node == "h100-9":
        # Verified existing MagicDNS peer and existing mgmt credentials; no key
        # copies or host-key overrides. Keep the head route for Slurm commands.
        head = setup + "env -u LD_LIBRARY_PATH ssh " + opts + " root@h100-9 " + shlex.quote(command)
    elif node == "h100-16":
        head = setup + "env -u LD_LIBRARY_PATH ssh " + opts + " h100-16 " + shlex.quote(command)
    else:
        head = setup + "env -u LD_LIBRARY_PATH ssh " + opts + " h100-16 " + shlex.quote(setup + worker)
    return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ProxyCommand=nc -X 5 -x 127.0.0.1:7892 %h %p", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3", "mgmt-host", head]

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--node",default="h100-9"); p.add_argument("--stdin"); p.add_argument("command"); a=p.parse_args()
    with open(a.stdin,"rb") if a.stdin else open("/dev/null","rb") as stream:
        sys.exit(subprocess.call(route(a.command,a.node),stdin=stream))
