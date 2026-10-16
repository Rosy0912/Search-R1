"""Bind trained weight bytes to the exact original training tokenizer."""
import os
import shutil
from pathlib import Path
from .contracts import sha,require

TOKENIZER_FILES={'tokenizer.json','tokenizer_config.json','special_tokens_map.json',
                 'vocab.json','merges.txt','added_tokens.json'}


def view_files(plan):
    files={n:h for n,h in plan['model_files'].items() if n not in TOKENIZER_FILES}
    actual=set(plan['canonical_tokenizer_files'])
    require({'tokenizer.json','tokenizer_config.json','vocab.json','merges.txt'}<=actual<=TOKENIZER_FILES,
            'Incomplete canonical tokenizer identity')
    files.update(plan['canonical_tokenizer_files'])
    return files


def validate_view(plan,expected_tokenizer_sha):
    require(plan['canonical_tokenizer_files']['tokenizer.json']==expected_tokenizer_sha,
            'Canonical tokenizer is not the archived training tokenizer')
    view=Path(plan['inference_model']).resolve(strict=True)
    canonical=Path(plan['canonical_tokenizer']).resolve(strict=True)
    weights=Path(plan['model']).resolve(strict=True)
    expected=view_files(plan)
    require({n for n in TOKENIZER_FILES if (canonical/n).is_file()}==set(plan['canonical_tokenizer_files']),
            'Canonical tokenizer file set differs')
    require({p.name for p in view.iterdir()}==set(expected),'Inference view has unexpected files')
    for name,digest in expected.items():
        p=view/name;source=(canonical if name in TOKENIZER_FILES else weights)/name
        require(p.is_file() and not p.is_symlink() and sha(p)==digest,'Inference view file differs: '+name)
        if name in TOKENIZER_FILES:
            require(not os.path.samefile(p,source),'Tokenizer must be an independent immutable copy: '+name)
        else:
            require(os.path.samefile(p,source),'Inference weights must hardlink their bound source file: '+name)
    return expected


def create_view(plan,expected_tokenizer_sha):
    canonical=Path(plan['canonical_tokenizer']).resolve(strict=True)
    weights=Path(plan['model']).resolve(strict=True)
    expected=view_files(plan)
    require(expected['tokenizer.json']==expected_tokenizer_sha,'Training tokenizer SHA differs')
    for name,digest in expected.items():
        source=(canonical if name in TOKENIZER_FILES else weights)/name
        require(source.is_file() and sha(source)==digest,'Source view identity differs: '+name)
    view=Path(plan['inference_model']);view.mkdir(parents=True,exist_ok=False)
    for name in expected:
        source=(canonical if name in TOKENIZER_FILES else weights)/name
        if name in TOKENIZER_FILES:
            # Linking/unlinking a live training tokenizer changes its ctime.
            # The training SHA guard correctly refuses a concurrent metadata
            # change; small independent copies avoid modifying that inode.
            with source.open('rb') as src,(view/name).open('xb') as dst:
                shutil.copyfileobj(src,dst,8*1024**2)
                dst.flush();os.fsync(dst.fileno())
        else:
            os.link(source,view/name)
    return validate_view(plan,expected_tokenizer_sha)
