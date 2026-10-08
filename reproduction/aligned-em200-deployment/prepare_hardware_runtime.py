"""Create the new hardware-only interpreter anchor, without starting GPU work."""
from pathlib import Path
import ast, datetime, hashlib, json, os, subprocess, sys
ROOT=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009')
SOURCE='080ebbc46b89d61567cd33e1a5e2f1b40afd8b84'
SUPERVISOR_SHA256={'resource_watch.py': '5ea7c46d2a8b4cda5e666cacb21b781a1d2a8908cd605950f3fae6c0d1774025', 'owned_processes.py': '2c8cdb3363461b402c3155091cab89739524256b1f89ab58c0f5e62334a877bd'}

def require(value,message):
    if not value:raise ValueError(message)

def digest(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
def write(path,raw):
    require(path.parent.resolve()==path.parent and not path.is_symlink(),'canonical own parent required')
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())

def main():
    require(os.uname().nodename=='h100-5' and sys.platform=='linux' and sys.prefix==str(ROOT/'env'),'actual new Linux environment required')
    require(ROOT.resolve()==ROOT,'canonical own ROOT required')
    require(subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()==SOURCE,'fixed source differs')
    require(not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True).strip(),'dirty production source')
    logs=ROOT/'logs';tools=ROOT/'tools';deployed=tools/'deployment'
    names=['resource_watch.py','owned_processes.py']
    records=[logs/'runtime-executable.json',logs/'tools-manifest.json',logs/'launch-approval.json',ROOT/'source-expected.txt']+[tools/n for n in names]
    require(all(not p.exists() and not p.is_symlink() for p in records),'existing/partial hardware preparation must be inspected, never overwritten')
    cpu_path=logs/'target-linux-cpu-compatible.json';cpu_log=logs/'target-linux-cpu-compatible.log'
    require(all(p.is_file() and not p.is_symlink() and p.resolve()==p for p in (cpu_path,cpu_log)), 'canonical actual CPU files required')
    cpu=json.loads(cpu_path.read_text())
    require(cpu['log']==str(cpu_log) and cpu['host']=='h100-5' and cpu['env_prefix']==str(ROOT/'env'),'CPU receipt task identity differs')
    actual=cpu['actual_python']
    require(actual['prefix']==str(ROOT/'env') and actual['executable']==str(ROOT/'env/bin/python') and actual['host']=='h100-5' and actual['platform']=='linux','CPU actual interpreter differs')
    require(cpu['scope']=='ACTUAL_TARGET_LINUX_CPU_WITH_EXPLICIT_TEST_ONLY_COMPATIBILITY_NOT_GPU_TRAINING','CPU scope differs')
    runner=deployed/'run-target-cpu-compatible.py'
    require(runner.resolve()==runner and digest(runner.read_bytes())=='7c9bfb09946629faa91581bd9e10c968bd3da7544bf6b9543a195c6ddf37170f','reviewed CPU runner differs')
    tree=ast.parse(runner.read_text())
    suite_names=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='names' for t in n.targets))
    expected=[str(ROOT/'env/bin/python'),'-B','-m','pytest','-q','-p','aligned_linux_cpu_plugin']+['tests/'+n for n in suite_names]
    require(len(suite_names)==23 and cpu['command']==expected,'full reviewed CPU suite differs')
    require(cpu['source']==SOURCE and cpu['exit']==0 and '255 passed' in cpu['tail'],'actual target CPU prerequisite missing')
    require(digest(cpu_log.read_bytes())==cpu['log_sha256']=='7eb44f559e071ef67a1ba035590753c71de0e914aeb03116f978ece716205cb0','independently reviewed actual CPU log changed')
    # Bind the actual interpreter inode/bytes; not a GPU, hardware, or train receipt.
    exe=Path('/proc/self/exe').resolve();st=exe.stat()
    record=dict(source_commit=SOURCE,root=str(ROOT),sys_prefix=sys.prefix,
        executable=dict(resolved_path=str(exe),device=st.st_dev,inode=st.st_ino),
        executable_sha256=digest(exe.read_bytes()),host=os.uname().nodename,
        scope='ACTUAL_LINUX_INTERPRETER_IDENTITY_NOT_GPU_OR_LAUNCH_ACCEPTANCE')
    raw=encoded(record)
    blobs={}
    for n in names:
        require((deployed/n).resolve()==deployed/n and not (deployed/n).is_symlink(),'canonical installed tool required')
        blobs[n]=(deployed/n).read_bytes()
        require(digest(blobs[n])==SUPERVISOR_SHA256[n],'independently reviewed supervisor bytes differ')
    files={n:digest(blobs[n]) for n in names}
    manifest=encoded(dict(source_commit=SOURCE,files=files,
        scope='HARDWARE_RUNTIME_SUPERVISOR_BYTES_NOT_TRAINING_ACCEPTANCE'))
    approval=dict(protocol='BAPO_ALIGNED_EM200_LAUNCH_V1',source_commit=SOURCE,root=str(ROOT),
        status='HARDWARE_ONLY_RUNTIME_ANCHOR_NOT_TRAINING_AUTHORIZATION',
        runtime_executable=dict(path=str(logs/'runtime-executable.json'),bytes=len(raw),sha256=digest(raw)),
        tools_manifest_sha256=digest(manifest),time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='NO_GPU_WORK_PERFORMED_NO_HARDWARE_RECEIPT_NO_TRAINING_AUTHORIZATION')
    for n in names:write(tools/n,blobs[n])
    write(ROOT/'source-expected.txt',(SOURCE+'\n').encode());write(logs/'runtime-executable.json',raw)
    write(logs/'tools-manifest.json',manifest);write(logs/'launch-approval.json',encoded(approval))
    sys.path.insert(0,str(tools))
    import resource_watch
    require(Path(resource_watch.__file__).resolve()==tools/'resource_watch.py','actual supervisor module differs')
    result=resource_watch.load_runtime_anchor(ROOT/'runs'/'hardware-anchor-validation',verify_current=True)
    print(json.dumps(dict(source=SOURCE,root=str(ROOT),runtime_anchor_sha256=result['sha256'],scope=approval['scope'])))

if __name__=='__main__':main()
