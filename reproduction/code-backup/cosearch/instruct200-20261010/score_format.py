"""Adapt format tag aliases only in a separate deployed scoring source."""
import ast
import hashlib
import json
from pathlib import Path


def patch(root):
    p=Path(root)/'verl/verl/utils/reward_score/search_qa_f1_with_format_penalty.py'
    before=p.read_text()
    tree=ast.parse(before)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_response_structure')
    assert isinstance(function.body[0],ast.Expr) and isinstance(function.body[0].value,ast.Constant)
    lines=before.splitlines(keepends=True)
    at=function.body[0].end_lineno
    lines[at:at]=['    from question_metadata import canonical_format_for_score\n',
                  '    processed_str = canonical_format_for_score(processed_str)\n']
    after=''.join(lines);ast.parse(after);p.write_text(after)
    receipt={'path':str(p),'before':hashlib.sha256(before.encode()).hexdigest(),'after':hashlib.sha256(after.encode()).hexdigest(),'scope':'format_tag_aliases_only'}
    (Path(root)/'score-format-patch.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
