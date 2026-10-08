"""Prepare original immutable datasets and common content-identified holdout."""
import json
import os
from pathlib import Path
import runpy
from hydra import compose, initialize_config_dir
from search_r1.baselines.holdout import prepare_common_holdout

root = Path(os.environ['AGENT_ROOT'])
repo = root/'repo'
runpy.run_path(str(repo/'reproduction/h20/prepare_paper_data.py'), run_name='__main__')
with initialize_config_dir(version_base=None, config_dir=str(repo/'verl/trainer/config')):
    config = compose(config_name='ppo_baselines')
manifest = prepare_common_holdout(config)
assert manifest['train_rows'] == 166222
assert manifest['holdout_rows'] == 3393
assert manifest['content_identity']['sha256'] == 'fac19c74fe28df2016bdff12f74f035fe272bed0d65d37e9740071e674195524'
(root/'logs/common-data-manifest.json').write_text(json.dumps(manifest, indent=2))
print('TOPD_COMMON_DATA_IDENTITY_PASS', manifest['content_identity']['sha256'], flush=True)
