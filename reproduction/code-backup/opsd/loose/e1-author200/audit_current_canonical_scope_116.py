#!/usr/bin/env python3
"""Local read-only audit of current pointers, not training/effect acceptance."""
import datetime
import hashlib
import json
import os
import pathlib
import stat

WORK = pathlib.Path(__file__).resolve().parent
BASE = WORK.parents[1]
CANONICAL = pathlib.Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/opsd')
SOURCE = '188f4e5e7b63cf479290d0888a4a075499886750'
CONFIG = '73f6401e818dc93a11c705da2357d0f3d91b2a51609a51e78a599fb7cf416c60'
PINS = {
    'progress100': ('ACTUAL100_PROGRESS_AND_FIXED_SET_EFFECT_INDEPENDENT_REVIEW.json', '4558b1bcd9a66e4ea20d1d8d9531db1c22d5038f1a56c9c2693a301d9692ad29'),
    'cpu100': ('ACTUAL100_CPU_STORED_CONTENT_INDEPENDENT_REVIEW.json', '2aaa49d7ea4d03653a849cc1cc40426a347d3e9fd4dcd12a951ed59e9de5174b'),
    'deployment': ('ACTUAL_FINAL_ARCHIVE_25E433_DEPLOYMENT_INDEPENDENT_REVIEW.json', 'b38ced8608b7f94536e0a6ce2fb214e62c78b1943c0b3ce76db063b1dfa6d7d1'),
    'driver116': ('live-driver-precheck125-20261006T013214224215Z/transport-original.json', '9fbf014910f86bfa11570a03244d45f094473e989155709a6005c04355d2a74f'),
    'status_before_root_fix': ('status-before-116-driver-and-scope-fix-20261006T013344434144Z-original.json', '3fb96f266f2fcb5a0e3d7314f944a1d067aaf450dd00ad857f11dc05586daeda'),
}


def stable_read(path):
    path = pathlib.Path(path)
    before = path.lstat()
    assert stat.S_ISREG(before.st_mode) and not path.is_symlink(), str(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_fd = os.fstat(fd)
    finally:
        os.close(fd)
    after_path = path.lstat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_uid', 'st_mode')
    identities = [tuple(getattr(s, k) for k in fields) for s in (before, opened, after_fd, after_path)]
    assert all(x == identities[0] for x in identities), 'File changed during read: ' + str(path)
    raw = b''.join(chunks)
    assert len(raw) == opened.st_size
    return raw, {'physical_path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def main():
    references = {}
    values = {}
    for key, (relative, expected) in PINS.items():
        raw, record = stable_read(WORK / relative)
        assert record['sha256'] == expected, 'Reference SHA differs: ' + key
        references[key] = record
        values[key] = json.loads(raw)

    status_raw, status_record = stable_read(CANONICAL / 'STATUS.json')
    cont_raw, cont_record = stable_read(BASE / 'work/CONTINUATION.md')
    report_raw, report_record = stable_read(CANONICAL / 'REPORT.md')
    status = json.loads(status_raw)
    author = status['author200']
    prep = author['final_evidence_preservation_preparation']
    collector = prep['collector_local_immutable_bundle']
    latest = author['latest_driver_only_observation']
    cont_head = cont_raw.decode().split('HISTORICAL HANDOFF RECORDS FOLLOW', 1)[0]
    report_head = report_raw.decode().split('HISTORICAL REPORT RECORDS FOLLOW', 1)[0]
    progress = values['progress100']
    cpu = values['cpu100']
    deployment = values['deployment']
    driver = values['driver116']

    assert status['current_experiment'] == 'author200' and status['gpu_job'] == 3261
    for key in ('current_completed_outer_iterations', 'completed_outer_iterations', 'current_source_actual_outer_iterations', 'actual_iterations'):
        assert type(status[key]) is int and status[key] == 100, key
    assert author['actual_outer'] == author['independently_reviewed_complete_outer'] == 100
    assert progress['source_commit'] == cpu['source_commit'] == deployment['source_commit'] == driver['source_commit'] == SOURCE
    assert progress['config_sha256'] == cpu['config_sha256'] == CONFIG
    assert progress['job_id'] == '3261' and cpu['training_job_id'] == '3261'
    assert progress['actual_complete_outer'] == progress['aggregate_complete_outer'] == cpu['checkpoint_step'] == 100
    assert progress['same_process_Adam_calls_per_rank'] == 1600
    assert author['latest_progress_independent_review_sha256'] == references['progress100']['sha256']
    inspection = author['checkpoint100_cpu_inspection']
    assert inspection['independent_review_sha256'] == references['cpu100']['sha256']
    assert inspection['native_report_sha256'] == cpu['native_report_sha256']
    assert inspection['full_native_content_acceptance'] is True
    assert cpu['scientific_effect_accepted'] is False
    assert cpu['whole_HF100_independent_load_accepted'] is False
    assert cpu['live_or_new_process_restore_accepted'] is False
    assert 'FIRST5_ONLY' in author['cpu_inspection_state'] and 'no wholeHF100' in author['cpu_inspection_state']

    assert latest['original_sha256'] == references['driver116']['sha256']
    assert latest['actual_complete_outer'] == driver['complete_counts']['driver'] == driver['latest_outer']['actual_outer_updates'] == 116
    assert latest['observed_utc'] == driver['utc']
    assert latest['full_independent_snapshot_still_outer'] == 100
    assert 'NOT_ALL_LOGS_OR125_CONTENT_ACCEPTANCE' in latest['scope']
    assert latest['batch_reward'] == driver['latest_outer']['metrics']['critic/rewards/mean']
    assert latest['nvml_mib'] == driver['nvml']
    assert driver['ready_for_one_cpu_inspection'] is False
    assert driver['existing_marker'] is False and driver['existing_output'] is False
    assert driver['allocation']['driver']['pid'] == author['driver_pid'] == 1533312
    assert driver['allocation']['driver']['start_ticks'] == author['driver_start_ticks'] == '1829085316'

    assert collector['deployed'] is True and collector['collector_executed'] is False
    assert collector['actual_deployment_returncode'] == deployment['actual_returncode'] == 0
    assert collector['deployment_independent_review_sha256'] == references['deployment']['sha256']
    assert collector['manifest_sha256'] == deployment['manifest_sha256']
    assert deployment['collector_executed'] is False and deployment['final_training_effect_or_resource_release_accepted'] is False
    assert 'ACTUALLY_DEPLOYED' in prep['new_collector_v2_software_review']['state']
    assert 'COLLECTOR_NOT_EXECUTED' in prep['new_collector_v2_software_review']['state']
    assert prep['final200_native_or_HF_launched'] is False
    assert status['final_51713_complete'] is False
    for key in ('target200_complete', 'final51713_complete', 'slurm_terminal_verified'):
        assert author[key] is False, key
    assert author['target_outer'] == 200
    assert datetime.datetime.fromisoformat(status['updated_at_utc']) >= datetime.datetime.fromisoformat(latest['observed_utc'])

    for head in (cont_head, report_head):
        assert '116' in head and '100' in head and '51713' in head
        assert 'HISTORICAL' not in head
    assert 'not full8rank116 acceptance' in cont_head
    assert 'wholeHF first5 only' in cont_head
    assert 'deployedb38 but notexecuted' in cont_head
    assert '尚未计入全八卡独审' in report_head
    assert '4.49%' in cont_head and '4.49%' in report_head
    assert '18.36%' in cont_head and '18.36%' in report_head

    # Historical scope ambiguity was identified before root changed the pointers.
    old = values['status_before_root_fix']['author200']
    assert old['final_evidence_preservation_preparation']['collector_local_immutable_bundle']['deployed'] is True
    assert old['final_evidence_preservation_preparation']['new_collector_v2_software_review']['state'] == 'SOFTWARE_ONLY_NOT_DEPLOYED_OR_FINAL_OBSERVATION'
    assert 'FIRST5_ONLY' not in old['cpu_inspection_state']
    assert old['latest_driver_only_observation']['actual_complete_outer'] == 111

    result = {
        'version': 1,
        'status': 'PASS_CURRENT_CANONICAL_POINTERS_AFTER_ROOT_SCOPE_FIX',
        'reviewer_role': 'READ_ONLY_CANONICAL_SCOPE_AUDITOR',
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'Current pointer consistency only; no new training/native/HF/effect/terminal/resource acceptance.',
        'source_commit': SOURCE,
        'training_job_id': 3261,
        'current_documents': {'status': status_record, 'continuation': cont_record, 'report': report_record},
        'current_authoritative_heads': {'continuation': cont_head, 'report': report_head},
        'references': references,
        'independently_accepted_complete_outer': 100,
        'latest_driver_only_complete_outer': 116,
        'first5_whole_HF_scope_preserved': True,
        'CPU100_stored_content_not_whole_HF_or_training_restore': True,
        'collector_software_deployed_but_not_executed': True,
        'actual125_or200_or_final51713_accepted': False,
        'historical_findings_corrected_by_root': [
            'Current collector software state said not deployed despite the adjacent actual deployment anchor.',
            'Generic CPU inspection state did not name first5 as the only whole-HF-load scope.',
            'Current handoff/report clocks and authority paragraph did not distinguish latest driver-only progress from accepted100.',
        ],
        'initial_inline_audit_attempt': {
            'actual_exit_code': 1,
            'message': 'ValueError: Actual reference SHA differs',
            'scope': 'The initial inline caller pinned driver111 while root advanced the current pointer to116. Fixed100/CPU100/deployment original SHA anchors remain unchanged. No PASS was saved for that failed attempt.'
        },
        'remaining_current_pointer_blockers': [],
        'limitations': [
            'A document consistency audit does not elevate driver-only116 to an all-rank independently accepted iteration.',
            'An older history section can contain superseded current/latest wording; each document has an explicit history delimiter.',
            'Negative fixed-set effect is preserved; no numeric/infrastructure PASS is recast as improved effect.',
            'Future125/200, wholeHF200, final51713, closed-lifetime logs, real exit and resource release remain unaccepted.',
        ],
        'canonical_edits_by_reviewer': False,
        'remote_or_GPU_execution_by_reviewer': False,
    }
    out = WORK / 'CURRENT_CANONICAL_116_SCOPE_FOLLOWUP_AUDIT.json'
    with out.open('x') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(json.dumps({'status': result['status'], 'report': str(out), 'sha256': hashlib.sha256(out.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
