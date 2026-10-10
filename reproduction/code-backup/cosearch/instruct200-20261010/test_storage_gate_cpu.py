"""Ensure unresolved storage approval cannot reach GPU acquisition."""
import json,tempfile
from pathlib import Path
from unittest.mock import patch
import run_allocation
with tempfile.TemporaryDirectory() as d:
    run_allocation.ROOT=Path(d)
    for record in ({'human_authorized':False,'budget_bytes':1500000000000},{'human_authorized':True,'budget_bytes':1000000000000}):
        (Path(d)/'storage-authorization.json').write_text(json.dumps(record))
        with patch.object(run_allocation,'acquire') as acquire:
            try:run_allocation.main()
            except RuntimeError as error:assert 'Storage authorization' in str(error)
            else:raise AssertionError('Unapproved storage must fail')
            acquire.assert_not_called()
print('PASS: pending or insufficient storage approval cannot reach GPU acquisition')
