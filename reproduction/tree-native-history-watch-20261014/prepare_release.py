"""Prepare a tiny code-only watcher packet locally; never uses SSH or systemctl."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

SOURCE = 'ba9f6627ec9a4e9749830d7b6acca53f6feec6b4'
TRAIN_UNIT = 'sst-tree-nested-branch-fresh200-ba9f6627.service'
WATCH_UNIT = 'sst-tree-native-history-watch-ba9f6627.service'
RUN = '/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:' + SOURCE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip():
        raise ValueError('Commit code before preparing watcher packet')
    output = args.output.resolve()
    if output == repo or output.is_relative_to(repo) or output.exists():
        raise ValueError('Use a new packet directory outside the source repo')
    runtime = repo / 'reproduction/tree-native-history-watch-20261014'
    names = ['watch_native_history.py', 'native_manifest_history_collector.py']
    expected = {}
    for name in names:
        path = runtime / name
        raw = path.read_bytes()
        # Require the actual published bytes, rather than silently packaging dirty files.
        committed = subprocess.check_output(['git', 'show', commit + ':reproduction/tree-native-history-watch-20261014/' + name], cwd=repo)
        if raw != committed or len(raw) > 200000:
            raise ValueError('Watcher packet code differs from small committed source: ' + name)
        expected[name] = hashlib.sha256(raw).hexdigest()
    output.mkdir(parents=True)
    for name in names:
        shutil.copy2(runtime / name, output / name)
    release = dict(schema='native_manifest_history_watch_v1', watcher_code_commit=commit,
                   training_source_commit=SOURCE, training_unit=TRAIN_UNIT, watcher_unit=WATCH_UNIT,
                   run_directory=RUN, interval_seconds=60, files=expected,
                   tensor_contents_read=False, PT_file_sha256_recomputed=False,
                   training_signals_sent=False, independent_cold_reload_performed=False)
    raw = (json.dumps(release, sort_keys=True, indent=2) + '\n').encode()
    release_sha = hashlib.sha256(raw).hexdigest()
    (output / 'release.json').write_bytes(raw)
    remote = '/dataset/shaoshitong/search-r1-rl/tree-grpo/tools/native-history-watch-' + commit
    history = RUN + '/output/native-manifest-history'
    service = '''[Unit]
Description=Owned native manifest JSON metadata audit (ba9f6627)
After=local-fs.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 "REMOTE/watch_native_history.py" --release "REMOTE/release.json" --release-sha256 RELEASE_SHA
WorkingDirectory=REMOTE
Restart=no
MemoryAccounting=yes
MemoryMax=256M
CPUAccounting=yes
CPUQuota=25%
TasksMax=16
RuntimeMaxSec=7d
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
RestrictAddressFamilies=AF_UNIX
ReadWritePaths="HISTORY"
Environment=PYTHONDONTWRITEBYTECODE=1
StandardOutput=journal
StandardError=journal
SyslogIdentifier=sst-native-history-ba9f6627
LogRateLimitIntervalSec=60s
LogRateLimitBurst=10

[Install]
WantedBy=multi-user.target
'''.replace('REMOTE', remote).replace('RELEASE_SHA', release_sha).replace('HISTORY', history)
    (output / WATCH_UNIT).write_text(service)
    receipt = dict(status='CODE_ONLY_WATCHER_PACKET_PREPARED_NOT_DEPLOYED', watcher_code_commit=commit,
                   training_source_commit_unchanged=SOURCE, remote_code_directory=remote,
                   watcher_unit=WATCH_UNIT, training_unit=TRAIN_UNIT, release_sha256=release_sha,
                   files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()},
                   actual_remote_deployment=False, actual_watcher_started=False,
                   training_signals_sent=False, independent_cold_reload_performed=False)
    (output / 'packet-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
