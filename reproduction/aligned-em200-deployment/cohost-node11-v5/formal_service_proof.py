"""Birth-bound formal services; evidence is for this job, never borrowed hardware PID."""
from pathlib import Path
import argparse, json, os, sys, time, subprocess, ipaddress
from formal_admission import ROOT, SOURCE, require, raw_file


def identity(pid):
    proc = Path('/proc') / str(pid)
    fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    require(fields[0] not in ('Z', 'X'), 'own service is not live')
    return dict(pid=pid, ppid=int(fields[1]), start_ticks=int(fields[19]), uid=proc.stat().st_uid)


def own_identity(pid):
    before = identity(pid)
    proc = Path('/proc') / str(pid)
    environment = (proc / 'environ').read_bytes().split(b'\0')
    for key in ('BAPO_RUN_ID', 'SLURM_JOB_ID'):
        require((key + '=' + os.environ[key]).encode() in environment, 'foreign service run/job')
    require(before['uid'] == os.getuid(), 'foreign service UID')
    executable = (proc / 'exe').resolve(); st = executable.stat()
    result = dict(before, command=[v.decode() for v in (proc / 'cmdline').read_bytes().split(b'\0') if v], executable=dict(path=str(executable),device=st.st_dev,inode=st.st_ino))
    require(identity(pid) == before, 'service birth changed during observation')
    return result


def retriever():
    import bind_retrieval_proof as binding
    require(Path(binding.__file__).resolve() == ROOT / 'tools/deployment/cohost-node11-v5/bind_retrieval_proof.py', 'foreign birth helper')
    return binding.birth(int(os.environ['BAPO_LOCAL_RETRIEVER_PID']), Path(os.environ['BAPO_RUN_ID']), os.environ['SLURM_JOB_ID'])


def port_owner(port):
    inodes = set()
    for name in ('tcp', 'tcp6'):
        for line in (Path('/proc/net') / name).read_text().splitlines()[1:]:
            fields=line.split()
            if fields[3] == '0A' and int(fields[1].rsplit(':',1)[1],16) == port:
                inodes.add(fields[9])
    require(len(inodes) == 1, 'own listener absent/ambiguous')
    inode=next(iter(inodes)); owners=set()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            for fd in (proc/'fd').iterdir():
                try:
                    if os.readlink(fd) == 'socket:['+inode+']': owners.add(int(proc.name))
                except OSError: pass
        except OSError: pass
    require(len(owners) == 1, 'listener owner unavailable/ambiguous')
    return own_identity(next(iter(owners)))


def ray_head():
    cli = own_identity(int(os.environ['BAPO_RAY_CLI_PID']))
    expected = [str(ROOT/'env/bin/python'), '-B', str(ROOT/'tools/deployment/cohost-node11-v5/ray_cli_compat.py'), 'start', '--head', '--block']
    require(cli['command'][:len(expected)] == expected and '--port=16383' in cli['command'], 'wrong actual Ray CLI')
    owner = port_owner(16383)
    require(Path(owner['executable']['path']).name == 'gcs_server', 'Ray head port is not own GCS')
    chain=[]; current=owner['pid']
    for _ in range(64):
        value=own_identity(current);chain.append(value)
        if current == cli['pid']: break
        current=value['ppid']; require(current > 1 and current not in {r['pid'] for r in chain}, 'Ray GCS is not a child of own CLI')
    else: raise ValueError('Ray head ancestry too deep')
    require(chain[-1] == cli, 'Ray CLI ancestry mismatch')
    require(all(own_identity(v['pid']) == v for v in chain), 'Ray head ancestry changed')
    require(port_owner(16383) == owner, 'Ray port changed during observation')
    return dict(cli=cli,listener=owner,chain=chain,address='127.0.0.1:16383')


def path(name):
    run=Path(os.environ['BAPO_RUN_ID']);require(run.parent==ROOT/'runs' and run.resolve()==run, 'canonical own formal service RUN')
    return run/name


def freeze(name, value):
    with path(name).open('x') as handle:
        json.dump(dict(source=SOURCE,run=os.environ['BAPO_RUN_ID'],job=os.environ['SLURM_JOB_ID'],identity=value,scope='ACTUAL_OWN_FORMAL_SERVICE_BIRTH_NOT_TRAINING_COMPLETION'),handle,sort_keys=True,indent=2);handle.flush();os.fsync(handle.fileno())


def compare(name, value):
    previous=json.loads(raw_file(path(name)))
    require(previous['source']==SOURCE and previous['run']==os.environ['BAPO_RUN_ID'] and previous['job']==os.environ['SLURM_JOB_ID'] and previous['identity']==value, 'formal service birth changed')


def validate_services():
    compare('own-formal-retriever-birth.json',retriever())
    compare('own-formal-ray-head.json',ray_head())


def active_local_ipv4():
    result=subprocess.run(['ip','-j','-4','address','show','up'],capture_output=True,text=True,timeout=10,check=True)
    rows=json.loads(result.stdout)
    require(isinstance(rows,list) and rows,'actual local interface inventory absent')
    addresses=set()
    for interface in rows:
        if 'UP' not in interface.get('flags',[]) or (interface.get('operstate')!='UP' and 'LOOPBACK' not in interface.get('flags',[])):continue
        for address in interface.get('addr_info',[]):
            if address.get('family')=='inet':
                ip=ipaddress.IPv4Address(address['local'])
                require(not ip.is_unspecified and not ip.is_multicast,'invalid local interface address')
                addresses.add(str(ip))
    require(addresses,'no active local IPv4')
    return dict(addresses=sorted(addresses),interfaces=rows)


def validate_local_ray_node(nodes, local):
    require(len(nodes)==1 and nodes[0].get('Resources',{}).get('GPU')==8,'connected Ray is not one eight-GPU node')
    ip=ipaddress.IPv4Address(nodes[0].get('NodeManagerAddress',''))
    require(str(ip) in local['addresses'],'Ray advertises a nonlocal/inactive interface address')
    require(type(nodes[0].get('NodeManagerPort')) is int and 0<nodes[0]['NodeManagerPort']<65536,'invalid own raylet port')
    return str(ip)


def verify_ray_connection():
    validate_services()
    import ray
    require(not ray.is_initialized(), 'unexpected preexisting driver Ray connection')
    ray.init(address='127.0.0.1:16383')
    try:
        nodes=[n for n in ray.nodes() if n.get('Alive')]
        local=active_local_ipv4()
        advertised_ip=validate_local_ray_node(nodes,local)
        raylet=port_owner(int(nodes[0]['NodeManagerPort']))
        require(Path(raylet['executable']['path']).name=='raylet', 'Ray node port is not own raylet')
        # Same marked job and CLI ancestry, also required for this node port.
        current=raylet['pid']; seen=set(); cli_pid=int(os.environ['BAPO_RAY_CLI_PID'])
        while current!=cli_pid:
            require(current>1 and current not in seen,'foreign Ray node ancestry');seen.add(current);current=own_identity(current)['ppid']
        own_identity(cli_pid)
        runtime=json.loads(raw_file(path('ray-runtime-path.json')))['path']
        session=Path(ray._private.worker.global_worker.node.address_info['session_dir'])
        require(session.parent==Path(runtime)/'ray' and session.resolve()==session,'foreign Ray session')
        require(port_owner(int(nodes[0]['NodeManagerPort']))==raylet,'own raylet changed during validation')
        require(advertised_ip in active_local_ipv4()['addresses'],'local Ray interface disappeared')
        validate_services()
        freeze('own-formal-ray-connection.json',dict(node=nodes[0],raylet=raylet,session_dir=str(session),address='127.0.0.1:16383',advertised_ip=advertised_ip,local_interface_evidence=local))
    finally:ray.shutdown()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['retriever-freeze','retriever-check','ray-freeze','check']);args=parser.parse_args()
    if args.action=='retriever-freeze':
        deadline=time.monotonic()+10
        while True:
            try:value=retriever();break
            except (OSError,ValueError):
                if time.monotonic()>=deadline:raise
                time.sleep(.05)
        freeze('own-formal-retriever-birth.json',value)
    elif args.action=='retriever-check':compare('own-formal-retriever-birth.json',retriever())
    elif args.action=='ray-freeze':freeze('own-formal-ray-head.json',ray_head())
    else:validate_services()


if __name__=='__main__':main()
