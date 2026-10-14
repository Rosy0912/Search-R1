"""Read physical evidence from explicit archived native continuation attempts.

An operator supplies hashes for entry and boundary archives in a chain file.
Training is never launched, stopped or reconfigured by this read-only audit.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .ablation_entry import TERMINAL
from .archive_entry import validate as validate_archive
from .inspect_run import attempt_evidence, checkpoint, read_jsonl, sha
from .resume_lineage import validate_link, merge_rounds


def load_bound(reference):
    path = Path(reference['path'])
    if path.is_symlink() or sha(path) != reference['sha256']:
        raise ValueError('Archived continuation evidence bytes changed')
    return json.loads(path.read_text())


def capture_boundary(run, manifest_path, expected_sha256):
    """Preserve manifest provenance before rolling cleanup; not a member audit."""
    run = Path(run).resolve()
    path = Path(manifest_path)
    if (not path.resolve().is_relative_to(run / 'checkpoints') or path.is_symlink()
            or sha(path) != expected_sha256):
        raise ValueError('Need this run’s exact explicit native checkpoint manifest')
    text = path.read_text()
    manifest = json.loads(text)
    record = json.loads((run / 'receipts/configuration.json').read_text())
    entry = json.loads((run / 'receipts/current-entry.json').read_text())
    identity = manifest['identity']
    if (identity['code_commit'] != record['source_commit']
            or identity['config_sha256'] != record['config_sha256']
            or identity['data_identity'] != record['data_identity']
            or str(identity['job_id']) != str(entry['allocation'])
            or manifest['status'] != 'COMPLETE_SAME_WORLD_STATE_SAVED'):
        raise ValueError('Native boundary is not from the current exact owned attempt')
    progress = [r for r in read_jsonl(Path(entry['output']) / 'completed-outers.jsonl')
                if r['completed_outer'] == identity['completed_step']]
    if (len(progress) != 1 or progress[0]['checkpoint']['path'] != str(path)
            or progress[0]['checkpoint']['sha256'] != expected_sha256
            or progress[0]['run_id'] != record['run_id']):
        raise ValueError('Manifest is not a fully recorded consumed-round boundary')
    value = dict(version=1, manifest_path=str(path), manifest_sha256=expected_sha256,
        manifest_original_text=text, completion_record=progress[0],
        scope='Manifest provenance only. Actual native member verification is required by restoration; this capture does not authorize a restart.')
    directory = run / 'receipts/native-boundary-history'
    directory.mkdir(exist_ok=True)
    target = directory / ('%06d-%s.json' % (identity['completed_step'], expected_sha256))
    try:
        with target.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
    except FileExistsError:
        if json.loads(target.read_text()) != value:
            raise ValueError('Existing immutable boundary archive disagrees')
    return dict(path=str(target), sha256=sha(target))


def read_segment(run, config, archive, scheduler):
    entry = archive['entry']
    entry_text = archive['entry_original_text']
    config_text = archive['configuration_original_text']
    if (hashlib.sha256(entry_text.encode()).hexdigest() != archive['entry_sha256']
            or json.loads(entry_text) != entry
            or hashlib.sha256(config_text.encode()).hexdigest() != archive['configuration_record_sha256']
            or json.loads(config_text) != config or archive['configuration'] != config):
        raise ValueError('Archived entry/configuration content does not match its frozen bytes')
    output = Path(entry['output'])
    if output.resolve() != output or output.is_symlink():
        raise ValueError('Physical attempt namespace changed')
    text = Path(archive['log']).read_text(errors='replace')
    state = validate_archive(config, entry, run, Path(config['config']).read_bytes(),
                             text, scheduler, entry['allocation'])
    identity = dict(job_id=str(entry['allocation']), run_id=config['run_id'],
        code_commit=config['source_commit'], config_sha256=config['config_sha256'],
        data_identity=config['data_identity'])
    fresh = json.loads((output / 'fresh-initialization.json').read_text())
    if any(fresh[k] != v for k, v in identity.items()):
        raise ValueError('Attempt initialization receipt belongs to another allocation/configuration')
    return dict(job=str(entry['allocation']), entry=entry, output=str(output), identity=identity,
        scheduler_state=state, log_text=text, attempts=attempt_evidence(run, output, identity, text))


def inspect_chain(run, chain_path, hash_native=False, query=None):
    run = Path(run).resolve()
    config = json.loads((run / 'receipts/configuration.json').read_text())
    chain = json.loads(Path(chain_path).read_text())
    if chain['version'] != 1 or chain['run_id'] != config['run_id'] or not chain['attempts']:
        raise ValueError('Explicit continuation chain does not identify the current run')
    archives = [load_bound(row['entry']) for row in chain['attempts']]
    jobs = [str(a['entry']['allocation']) for a in archives]
    if len(set(jobs)) != len(jobs):
        raise ValueError('Continuation chain repeats an allocation')
    current = json.loads((run / 'receipts/current-entry.json').read_text())
    if archives[-1]['entry'] != current:
        raise ValueError('Continuation chain does not end at the current actual entry')
    if query is None:
        query = lambda ids: subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', ','.join(ids),
            '--format=JobIDRaw,User,State,ExitCode'], text=True)
    scheduler = query(jobs)
    segments = [read_segment(run, config, a, scheduler) for a in archives]
    links = []
    for index, segment in enumerate(segments):
        if index == 0:
            if segment['entry'].get('resume_manifest') or chain['attempts'][index].get('parent_boundary'):
                raise ValueError('The first archived attempt must establish fresh initialization')
            continue
        parent = segments[index - 1]
        if parent['scheduler_state'].split()[0] not in TERMINAL:
            raise ValueError('A live or unknown parent allocation cannot establish a continuation')
        archived = load_bound(chain['attempts'][index]['parent_boundary'])
        manifest_text = archived['manifest_original_text']
        if hashlib.sha256(manifest_text.encode()).hexdigest() != archived['manifest_sha256']:
            raise ValueError('Archived parent manifest bytes changed')
        manifest = json.loads(manifest_text)
        progress = archived['completion_record']
        if (manifest['identity']['data_identity'] != config['data_identity']
                or progress['run_id'] != config['run_id']
                or progress['source_commit'] != config['source_commit']
                or progress['config_sha256'] != config['config_sha256']
                or progress['completed_outer'] != manifest['identity']['completed_step']
                or progress['checkpoint']['path'] != archived['manifest_path']
                or progress['checkpoint']['sha256'] != archived['manifest_sha256']):
            raise ValueError('Saved completion record and restored parent boundary disagree')
        physical_progress = read_jsonl(Path(parent['output']) / 'completed-outers.jsonl')
        if progress not in physical_progress:
            raise ValueError('Archived boundary lacks the actual parent completed-round record')
        restored = json.loads((Path(segment['output']) / 'native-resumed.json').read_text())
        link = validate_link(parent['entry'], segment['entry'], manifest, restored,
                             archived['manifest_path'], archived['manifest_sha256'])
        if progress['optimizer_calls_per_rank'] != link['actual_adam_per_rank']:
            raise ValueError('Restored Adam total differs from the actual saved progress')
        links.append(link)
    merged = merge_rounds(segments, links)
    milestones = {}
    for step in (5, 50, 100, 150):
        row = next((r for r in merged['attempts'] if r['outer'] == step), None)
        if row:
            owner = next(s for s in segments if s['job'] == row['allocation'])
            milestones[str(step)] = {k: owner[k] for k in ('job', 'output', 'identity')}
    native5 = (checkpoint(run, milestones['5']['identity'], hash_native) if '5' in milestones else
        dict(pre_evaluation_saved=False, post_evaluation_saved=False,
             same_world_reload_verified=False, all_native_files_hash_verified=False))
    sixth = []
    for index, segment in enumerate(segments):
        start = links[index - 1]['next_outer'] if index else 1
        stop = links[index]['completed_outer'] if index < len(links) else float('inf')
        if not start <= 6 <= stop:
            continue
        for line in segment['log_text'].splitlines(keepends=True):
            if not line.endswith('\n') or 'TREE_TIMING=' not in line:
                continue
            event = json.JSONDecoder().raw_decode(line.split('TREE_TIMING=', 1)[1])[0]
            if (event.get('run_id') == config['run_id'] and event.get('outer') == 6
                    and event.get('event') in ('generation_rpc_complete', 'trajectory_complete')):
                sixth.append(event)
    first5 = merged['attempts'][:5]
    finite = len(first5) == 5 and all(r['reported_pg_loss_finite'] is True and
        all(rank['all_observed_loss_finite'] is not False for rank in r['ranks']) for r in first5)
    writes = len(first5) == 5 and all(r['actual_adam_writes_per_rank'] > 0 for r in first5)
    return dict(at=time.time(), run=str(run), output=segments[-1]['output'], identity=segments[-1]['identity'],
        **merged, native5=native5, sixth_real_generation_observed=bool(sixth),
        sixth_first_evidence=sixth[0] if sixth else None, first5_observed_loss_all_finite=finite,
        first5_each_has_real_adam_writes=writes, zero_write_rounds=sum(r['actual_adam_writes_per_rank']==0 for r in merged['attempts']),
        initial_acceptance_verified=bool(finite and writes and native5['same_world_reload_verified']
            and native5['all_native_files_hash_verified'] and sixth),
        milestone_attempts=milestones, restoration_links=links, chain_sha256=sha(chain_path))
