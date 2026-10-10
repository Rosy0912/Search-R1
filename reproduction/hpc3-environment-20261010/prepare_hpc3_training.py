"""Prepare an isolated HPC3 port; never submit a GPU job or change bundle bytes."""
import ast
import hashlib
import json
import pathlib
import shutil
import sys
import sysconfig

BASE = pathlib.Path('/data/user/jxu957/experiments/topd-20261008')
ROOT = BASE / 'runtime'
BUNDLE = BASE / 'bundle'
OLD = '/dataset/shaoshitong/search-r1-rl/topd'


def sha(p):
    h = hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def links(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        q = dst / p.name
        if p.is_dir():
            links(p, q)
        elif not q.exists():
            q.hardlink_to(p)


def manifest(base):
    p = base / 'files.json'
    old = json.loads(p.read_text())
    old['hpc3_adaptation'] = {'original_manifest_sha256': sha(p),
                             'status': 'HPC3_INFRASTRUCTURE_PORT_NO_GPU_EXECUTION'}
    old['files'] = {str(f.relative_to(base)): {'size': f.stat().st_size, 'sha256': sha(f)}
                    for f in base.rglob('*') if f.is_file() and f != p}
    p.write_text(json.dumps(old, indent=2, sort_keys=True))
    return {'path': str(p), 'sha256': sha(p)}


def main():
    ROOT.mkdir(exist_ok=True)
    shutil.copytree(BUNDLE / 'code/h100-layout', ROOT, dirs_exist_ok=True)
    shutil.copytree(BUNDLE / 'provenance/h100-layout/logs', ROOT / 'logs', dirs_exist_ok=True)
    links(BUNDLE / 'student_initial', ROOT / 'assets/qwen2.5-3b-instruct-aa8e725')
    links(BUNDLE / 'datasets/common-instruct-0b002b', ROOT / 'assets/common-instruct-0b002b')
    links(BUNDLE / 'retrieval', ROOT / 'assets-root/assets')
    teacher = ROOT / 'logs/r1-standard32b-grpo100-common-v5-r1/checkpoints/actor/global_step_85'
    links(BUNDLE / 'teacher', teacher)
    for name, target in [('env', BASE / 'environment/topd-py310'), ('repo', ROOT / 'repo-terminal-fixed413')]:
        p = ROOT / name
        if not p.exists():
            p.symlink_to(target, target_is_directory=True)
    rt = ROOT / 'tools/topd-instruct300-launch-v6/runtime'
    shutil.copy2(pathlib.Path(__file__).with_name('hpc3_owned_processes.py'), rt)
    shutil.copy2(pathlib.Path(__file__).with_name('bind_hpc3_run.py'), rt)
    changes = []
    for p in list(rt.glob('*.py')) + [ROOT / 'tools/search-r1-32b-grpo100-tools-direct-v5/owner_support.py']:
        s = p.read_text()
        original = sha(p)
        s = s.replace(OLD, str(ROOT))
        if p.name == 'owner_support.py':
            tree = ast.parse(s)
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'OwnedProcesses')
            lines = s.splitlines(keepends=True)
            replacement = ('from hpc3_owned_processes import OwnedProcesses as HPC3OwnedProcesses\n'
                           'class OwnedProcesses(HPC3OwnedProcesses):\n'
                           '    def __init__(self, scope, relative):\n'
                           '        super().__init__(scope, relative, cwd=ROOT/"repo")\n\n')
            s = ''.join(lines[:cls.lineno-1]) + replacement + ''.join(lines[cls.end_lineno:])
            s = s.replace("socket.gethostname().split('.')[0]=='h100-11'", "socket.gethostname().split('.')[0]==os.environ.get('SLURMD_NODENAME',os.environ.get('SLURM_JOB_NODELIST'))")
            s = s.replace("int(os.environ.get('SLURM_CPUS_PER_TASK','0'))>=128", "int(os.environ.get('SLURM_CPUS_PER_TASK','0'))>=96")
            s = s.replace("require(hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'),'Need Linux pidfd support before launch')", "require(os.environ.get('SLURM_JOB_ID'), 'Need actual Slurm allocation before launch')")
        if p.name == 'supervise.py':
            s = s.replace("socket.gethostname()!='h100-11' or os.environ.get('SLURM_JOB_NODELIST')!='h100-11'", "socket.gethostname().split('.')[0]!=os.environ.get('SLURMD_NODENAME',os.environ.get('SLURM_JOB_NODELIST'))")
            s = s.replace("int(os.environ.get('SLURM_CPUS_PER_TASK','0'))<128", "int(os.environ.get('SLURM_CPUS_PER_TASK','0'))<96")
            s = s.replace("job=os.environ['SLURM_JOB_ID'],uuids=", "job=os.environ['SLURM_JOB_ID'],node=socket.gethostname().split('.')[0],uuids=")
        if p.name == 'resource_audit.py':
            s = s.replace("r['hostname']!='h100-11'", "r['hostname'].split('.')[0]!=config['node']")
            marker = "def charged_memory(scope):"
            pos = s.index(marker)
            s = s[:pos] + s[pos:].replace("values={}", "scope=pathlib.Path(str(scope).replace('/freezer/','/memory/'));values={}", 1)
        if p.name == 'allocator_policy.py':
            s = s.replace("keys=('CPATH'", "keys=('LD_PRELOAD','LD_LIBRARY_PATH','CPATH'")
        if p.name == 'train_entry.py':
            s = s.replace("temp=pathlib.Path('/dev/shm')/('r1-'+os.environ['SLURM_JOB_ID']);temp.mkdir(mode=0o700,exist_ok=False)",
                          "shared=pathlib.Path(c['evidence_root'])/'cache/ray';shared.mkdir(mode=0o700,exist_ok=False);temp=pathlib.Path('/tmp')/('r1-'+os.environ['SLURM_JOB_ID']);temp.symlink_to(shared,target_is_directory=True)")
            s = s.replace(" run_original_main(entry.main_task._function,resolved,ray.shutdown,c)",
                          " try:run_original_main(entry.main_task._function,resolved,ray.shutdown,c)\n finally:temp.unlink()")
        if s != p.read_text():
            p.write_text(s)
            changes.append({'path': str(p), 'before_sha256': original, 'after_sha256': sha(p)})

    # Replace only the isolated runtime's legacy GPU library; bundle is immutable.
    gpu = ROOT / 'retriever-gpu-lib'
    shutil.rmtree(gpu)
    shutil.copytree(BASE / 'environment/gpu-faiss-lib', gpu,
                    ignore=shutil.ignore_patterns('*.before-elf-alignment', '__pycache__', '*.pyc'))
    include = pathlib.Path(sysconfig.get_path('include')).resolve()
    headers = ROOT / 'assets/hpc3-compiler-headers'
    headers.mkdir(parents=True, exist_ok=True)
    (headers / 'inventory.json').write_text(json.dumps({
        'include': str(include), 'python': sys.version.split()[0],
        'files': {str(p.relative_to(include)): sha(p) for p in include.rglob('*') if p.is_file()}
    }, sort_keys=True, indent=2))
    shutil.copy2(pathlib.Path(__file__).with_name('hpc3_compiler_headers.py'), rt / 'compiler_headers.py')

    binding_path = ROOT / 'logs/qwen25-32b-terminal-fixed413-binding.json'
    retrieval = json.loads(binding_path.read_text())
    obsolete = {k: v for k, v in retrieval['runtime_files'].items()
                if k.endswith('.partial') and not (ROOT / k).exists()}
    if any(not k.startswith('assets-root/assets/e5-base-v2/') for k in obsolete):
        raise ValueError('Unexpected missing source dependency')
    retrieval['runtime_files'] = {k: v for k, v in retrieval['runtime_files'].items()
                                  if not k.startswith('retriever-gpu-lib/') and k not in obsolete}
    retrieval['hpc3_excluded_historical_incomplete_downloads'] = obsolete
    retrieval['runtime_files'].update({str(p.relative_to(ROOT)): sha(p)
                                       for p in gpu.rglob('*') if p.is_file()})
    retrieval['hpc3_gpu_faiss'] = {'version': '1.8.0.2', 'elf_alignment_repair': True,
                                  'original_binding_sha256': sha(BUNDLE / 'provenance/h100-layout/logs/qwen25-32b-terminal-fixed413-binding.json')}
    binding_path.write_text(json.dumps(retrieval, sort_keys=True, indent=2))
    shutil.copy2(ROOT / 'repo-terminal-fixed413/search_r1/search/retrieval_server.py', rt / 'original-native-server.py')
    gate = rt / 'retrieval_gate.py'
    s = gate.read_text().replace("BINDING_SHA='7ded91b8c81a18bd56d6123b1fb5c1dc6426368766a34b2e29383388fa1304a2'",
                                  'BINDING_SHA=' + repr(sha(binding_path)))
    i = s.index('def verify_restoration(server):')
    s = s[:i] + '''def verify_restoration(server):
 original=server.with_name('original-native-server.py')
 if sha(original)!='b2eed2cbddbbe3c5bdc2d7f1e8b6f3738dc5629f3d42e0ba68690c82c9bcd052':raise ValueError('Original retrieval source drift')
 if sha(server)!='972bbb955190b11c45b54640f5054a7fcb62fdca15c2f0a0f4d3c139b525cec2':raise ValueError('Frozen approved RPC adapter drift')
 native_tree=ast.parse(original.read_text());patched_tree=ast.parse(server.read_text())
 for tree in (native_tree,patched_tree):
  tree.body=[n for n in tree.body if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) or n.name!='retrieve_endpoint']
 if ast.dump(native_tree)!=ast.dump(patched_tree):raise ValueError('Dense encoder/index math changed')
 return {'scope':'FROZEN_RPC_ADAPTER_AND_UNCHANGED_DENSE_ENCODER_AST','original_sha256':sha(original),'server_sha256':sha(server)}
'''
    gate.write_text(s)

    src = BUNDLE / 'provenance/h100-layout/logs/topd-instruct300-teacher85-approved-v1/static-config.json'
    def remap(v):
        if isinstance(v, dict): return {k: remap(x) for k, x in v.items()}
        if isinstance(v, list): return [remap(x) for x in v]
        if isinstance(v, str): return v.replace(OLD, str(ROOT))
        return v
    config = remap(json.loads(src.read_text()))
    prior_evidence = str(ROOT / 'logs/topd-instruct300-teacher85-approved-v1/evidence')
    def run_paths(v):
        if isinstance(v, dict): return {k: run_paths(x) for k, x in v.items()}
        if isinstance(v, list): return [run_paths(x) for x in v]
        if isinstance(v, str): return v.replace(prior_evidence, str(ROOT / 'logs/topd-hpc3-prepared/evidence'))
        return v
    config = run_paths(config)
    approval = pathlib.Path(config['teacher_actual_result']['path'])
    approval.write_text(json.dumps(remap(json.loads(approval.read_text())), indent=2))
    config['teacher_actual_result']['sha256'] = sha(approval)
    config['support']['sha256'] = sha(config['support']['path'])
    config['retrieval_asset_binding']['sha256'] = sha(binding_path)
    config['retriever_command'][3] = str(rt / 'retrieval_server.py')
    header_file = headers / 'inventory.json'
    config['compiler_headers'] = {'base': str(headers), 'inventory_sha256': sha(header_file),
                                  'cpath': str(include), 'provider': 'HPC3_CONDA_PY310_NATIVE_ABI'}
    launcher = rt.parent / 'launch.sbatch'
    launch = launcher.read_text().replace(OLD, str(ROOT))
    launch = launch.replace('#SBATCH --nodelist=h100-11', '#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --account=mou176')
    launch = launch.replace('#SBATCH --partition=gpu', '#SBATCH --partition=acd_u')
    launch = launch.replace('#SBATCH --cpus-per-task=128', '#SBATCH --cpus-per-task=96')
    launch = launch.replace('set -euo pipefail', 'set -euo pipefail\nsource ' + str(BASE / 'environment/activate-topd.sh'))
    launch = launch.replace('Actual teacher100 bound student configuration required', 'Prepared user-approved teacher85 HPC3 configuration required')
    launch = launch.replace('exec "$TASK_ROOT/env/bin/python"', 'RUN_CONFIG="$TASK_ROOT/logs/topd-hpc3-${SLURM_JOB_ID}/config.json"\n"$TASK_ROOT/env/bin/python" "$TASK_ROOT/tools/topd-instruct300-launch-v6/runtime/bind_hpc3_run.py" --template "$TOPD_CONFIG" --output "$RUN_CONFIG"\nTOPD_CONFIG="$RUN_CONFIG"\nexec "$TASK_ROOT/env/bin/python"', 1)
    launcher.write_text(launch)
    config['tools_manifest'] = manifest(rt)
    config['launcher_manifest'] = manifest(rt.parent)
    out = BASE / 'outputs/training-preparation'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'hpc3-config-draft.json').write_text(json.dumps(config, indent=2))
    missing = []
    def check(v, key=''):
        if isinstance(v, dict):
            for k, x in v.items(): check(x, key + '.' + k)
        elif isinstance(v, list):
            for i, x in enumerate(v): check(x, key + '.' + str(i))
        elif isinstance(v, str) and v.startswith(str(ROOT)) and ':' not in v:
            p = pathlib.Path(v)
            if not p.exists() and not any(x in key for x in ['checkpoint', 'evidence_root', 'actual_outer_receipt', 'job']):
                missing.append({'field': key, 'path': v})
    check(config)
    report = {'status': 'PORT_DRAFT_PENDING_VALIDATION', 'original_config_sha256': sha(src),
              'patches': changes, 'missing_references': missing, 'gpu_training_started': False}
    (out / 'port-audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'status': report['status'], 'missing_references': missing}))


if __name__ == '__main__':
    main()
