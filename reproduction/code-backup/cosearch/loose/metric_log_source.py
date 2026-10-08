import hashlib, json
from pathlib import Path

def validated_metric_log(run, root):
    text=(run/'training.log').read_text(errors='replace')
    metric_source='training.log'
    if (run/'worker-training.log').is_file():
        provenance=json.loads((run/'worker-log-provenance.json').read_text())
        launch_for_log=json.loads((run/'launch.json').read_text())
        assert str(provenance['job'])==launch_for_log['job_id']
        source_path=Path(provenance['source']).resolve()
        assert source_path.is_relative_to(root.resolve()) and source_path.name.endswith('-'+str(provenance['worker_pid'])+'.out')
        assert Path(provenance['destination']).resolve()==run/'worker-training.log'
        worker_bytes=(run/'worker-training.log').read_bytes()
        assert len(worker_bytes)==provenance['bytes']
        assert hashlib.sha256(worker_bytes).hexdigest()==provenance['sha256']
        assert provenance['process_cgroup_verified'] and provenance['process_start_ticks']>0
        assert '/job_'+str(provenance['job'])+'/' in provenance['cgroup']
        assert Path(provenance['session']).resolve()==source_path.parent.parent
        source_stat=source_path.stat()
        assert source_stat.st_ino==provenance['source_inode'] and source_stat.st_dev==provenance['source_device']
        assert provenance['live_binding_at']<=provenance['captured_at']
        text=worker_bytes.decode(errors='replace')
        metric_source='worker-training.log'
    return text, metric_source
