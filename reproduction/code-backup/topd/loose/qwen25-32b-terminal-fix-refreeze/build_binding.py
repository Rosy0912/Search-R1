"""Create complete official32B evaluation binding, CPU/offline only; no submit/download."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import eval64

ROOT = Path('/dataset/shaoshitong/search-r1-rl/topd')
COMMIT = '413a10c50f32febaafc83db8170e2b005e0e979f'
LAUNCHER_SHA = 'd4de70880869abbec64f81ccd86f6f8dd8755b01d9db2cb93e2ec112d3031a09'
SOURCES = {
    'search_r1/qa_protocol.py': '9a825a7b222b5a3fd2c1de7f63ed38e157258fcf858e773ff64c963a4b6daddd',
    'search_r1/baselines/bapo.py': 'f95eebe667b02b2bab685cf2c62c6aabafb6021f7b9824e839d878876cdb7b94',
    'search_r1/baselines/evaluation.py': 'b7fd01f2993b22d475dd220b80ab9dcf225971bca98ecbe9016c271edcca145f',
    'search_r1/baselines/preflight.py': 'cc9445265a3835e1623c0de9d95380717f0e44d4a303d45e3fe1f559431e1605',
    'search_r1/baselines/topd_local_gpu.py': '4861cb7cd20fe9277a89dca21eff77e7e6107d91940e0f6591b6b75d3e5f6a89',
    'search_r1/llm_agent/generation.py': '91e281dc853c077dfcda3f05b64a714c009c8993fe05b2069fcc2def8f114470',
    'search_r1/llm_agent/tensor_helper.py': 'b000f3b61c5a2be13dfcf64e1ce9091ce89dda99ed3b331fb24ab1f43b4dce69',
    'search_r1/search/retrieval_server.py': 'b2eed2cbddbbe3c5bdc2d7f1e8b6f3738dc5629f3d42e0ba68690c82c9bcd052',
    'verl/protocol.py': 'ceccbe7c9ad3bfe430355f35705054881f32dc254b0862a1f0bf6800e387405d',
    'verl/trainer/main_ppo.py': '95e630c73fc0701738c28f48481ee338bd08aff771b513490260b61c412051b9',
    'verl/utils/dataset/rl_dataset.py': 'e95b34a590956ed37e69631c87712c9dacbddc0fae632ca1e5403ed19e409815',
    'verl/utils/reward_score/qa_em.py': 'fa5e8f9386b0db2411dafa7612bc633a1ced362232c81f2495fe78c7ed1b4997',
    'verl/utils/tokenizer.py': '839dd008294614789194d3d97bedbd2ae381038fc288d7ed1f600bca7dc3470b',
    'verl/utils/torch_functional.py': '21eedf934c643b6c0a6809df89915c17b80796254920dcd4cbe2ec5850a7f575',
    'verl/workers/rollout/vllm_rollout/vllm_rollout.py': 'da4a0338a8bd7144fc387d19d0fdae421f66bdbae1c627b2097b24eb2871512f',
}
TOOLS = {'eval64.py': '24d144fdac16b0983a11e3bd99d111f44d0bc312dc98b74847f3d16524076bee', 'supervisor.py': 'd79fc9dc99894ebc62d3a68ac1e12db16d100d6de46428ae84ec995229cb9964', 'runtime_safety.py': 'c7b16f1dd5d7cc133e55f034025c7dc8181ddcafaf59bdccfda02ceaf51f081d', 'test_eval64.py': '1bef8cf898202257556db74d7aadddbf513b59d37ac3218e3d1cb62c10e31f32', 'eval64.sbatch': '21aa558e8f6ce0ad3e719b56c91e8c32644add9dc28326400e1d3655e00bf0ba', 'native-cpu.sbatch': '2e0627622655e4085694acab27c1f7dd2d5be5cf1634bbf94a38ac3d6d6f0b14', 'test_terminal_quality.py': '42de20937f5a5da55310594bb4d59780aa30fee8a90faf3133bbba467db5d047'}


def local(path, root):
    path = Path(path).resolve()
    eval64.require(root in path.parents, 'File must resolve under own root: ' + str(path))
    return path


def signature(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def stable_sha(path, observations):
    before = signature(path)
    digest = eval64.sha(path)
    eval64.require(signature(path) == before, 'File changed while hashing: ' + str(path))
    observations[str(path)] = before
    return digest


def model_inventory(model, metadata, observations):
    required = {'config.json', 'generation_config.json', 'merges.txt', 'vocab.json',
                'tokenizer.json', 'tokenizer_config.json', 'model.safetensors.index.json'}
    official = {row['rfilename']: row for row in metadata['siblings']}
    shards = {name for name in official if name.endswith('.safetensors') and 'lfs' in official[name]}
    eval64.require(len(shards) == 17, 'Official inventory must include17 LFS shards')
    names = required | shards
    # Reject partial before streaming65GiB, even if later files might arrive.
    eval64.require(all((model / name).is_file() for name in names), 'Partial download: require all17 final shards and metadata')
    eval64.require(not any(p.is_file() and p.name.endswith(('.partial', '.part', '.tmp', '.incomplete')) for p in model.rglob('*')),
                   'Model directory still contains partial transfer artifacts')
    files = {}
    for name in sorted(names):
        path = local(model / name, model)
        expected = official[name]
        eval64.require(path.stat().st_size == expected['size'], 'Official file incomplete/wrong size: ' + name)
        files[name] = stable_sha(path, observations)
        if 'lfs' in expected:
            eval64.require(files[name] == expected['lfs']['sha256'], 'Official LFS checksum mismatch: ' + name)
        else:
            raw = path.read_bytes()
            blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
            eval64.require(blob == expected['blobId'], 'Official Git blob mismatch: ' + name)
    return files


def create_binding(root, model, metadata_path, data, output):
    root = Path(root).resolve()
    eval64.require(root == ROOT, 'Unexpected own root')
    model, metadata_path, data, output = [local(p, root) for p in (model, metadata_path, data, output)]
    eval64.require(root / 'logs' in output.parents, 'Binding must be exclusive own logs artifact')
    eval64.require(not output.exists() and not output.is_symlink(), 'Refuse existing binding output')
    eval64.require(output.parent.is_dir(), 'Output directory must already exist')
    observed = {}
    eval64.require(stable_sha(metadata_path, observed) == eval64.OFFICIAL_METADATA_SHA, 'Official captured metadata drift')
    metadata = json.loads(metadata_path.read_text())
    eval64.require(metadata['id'] == 'Qwen/Qwen2.5-32B' and metadata['sha'] == eval64.REVISION, 'Wrong official model/revision')
    head = subprocess.check_output(['git', '-C', str(root / 'repo-terminal-fixed413'), 'rev-parse', 'HEAD'], text=True).strip()
    eval64.require(head == COMMIT, 'Fixed production source HEAD drift')
    eval64.require(set(SOURCES) == eval64.REUSED_SOURCES, 'Builder source scope differs from evaluator')
    for name, expected in SOURCES.items():
        eval64.require(stable_sha(local(root / 'repo-terminal-fixed413' / name, root), observed) == expected, 'Frozen source drift: ' + name)
    for name, expected in TOOLS.items():
        eval64.require(stable_sha(Path(__file__).parent / name, observed) == expected, 'Frozen evaluation tool drift: ' + name)
    eval64.require(stable_sha(local(root / 'tools/topd-short-run.py', root), observed) == LAUNCHER_SHA, 'Frozen safety launcher drift')
    eval64.require(stable_sha(data, observed) == eval64.DATA_SHA, 'Original64 data drift')
    # No reliance on downloader PID/status or a self-declared complete flag.
    model_files = model_inventory(model, metadata, observed)
    runtime_files = {}
    fixed = ['assets-root/assets/wiki/e5_Flat.index', 'assets-root/assets/wiki/wiki-18.jsonl']
    for directory in ('assets-root/assets/e5-base-v2', 'retriever-gpu-lib'):
        folder = local(root / directory, root)
        eval64.require(folder.is_dir(), 'Missing real retrieval dependency: ' + directory)
        fixed.extend(str(p.relative_to(root)) for p in folder.rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and '.cache' not in p.parts)
    for name in sorted(set(fixed)):
        runtime_files[name] = stable_sha(local(root / name, root), observed)
    binding = dict(status='INVENTORY_VERIFIED', revision=eval64.REVISION,
                   official_metadata_path=str(metadata_path), official_metadata_sha256=eval64.OFFICIAL_METADATA_SHA,
                   model_directory=str(model), model_files=model_files, data_path=str(data), data_sha256=eval64.DATA_SHA,
                   code_commit=COMMIT, source_files=SOURCES, tool_files=TOOLS,
                   owned_launcher_sha256=LAUNCHER_SHA, runtime_files=runtime_files,
                   producer_scope='CPU_ONLY_COMPLETE_OFFICIAL_INVENTORY_NOT_GPU_OR_QUALITY_APPROVAL',
                   builder_sha256=eval64.sha(Path(__file__)))
    # Reuse unchanged evaluator's full header/config/source/runtime gate; only a
    # temporary candidate is written before acceptance, never the final binding.
    fd, temporary = tempfile.mkstemp(prefix='.candidate-binding-', suffix='.json', dir=output.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(binding, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        eval64.asset_gate(temporary, root)
        for name, expected in observed.items():
            eval64.require(signature(name) == expected, 'Dependency changed during full gate: ' + name)
        current_head = subprocess.check_output(['git', '-C', str(root / 'repo-terminal-fixed413'), 'rev-parse', 'HEAD'], text=True).strip()
        eval64.require(current_head == COMMIT, 'Fixed source HEAD changed during full gate')
        # Atomic exclusive publication of the already-fsynced complete JSON.
        # link fails with EEXIST and never replaces another producer's output.
        os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return binding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--model', default=str(ROOT / 'assets-root/assets/qwen2.5-32b'))
    parser.add_argument('--official-metadata', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    create_binding(args.root, args.model, args.official_metadata, args.data, args.output)
    print('CPU_COMPLETE_OFFICIAL32B_BINDING_CREATED_NO_GPU_NO_QUALITY_APPROVAL')


if __name__ == '__main__':
    main()
