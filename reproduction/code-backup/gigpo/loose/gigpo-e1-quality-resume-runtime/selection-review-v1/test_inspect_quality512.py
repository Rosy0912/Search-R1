import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('inspector',Path(__file__).with_name('inspect_quality512_v1.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

@pytest.mark.parametrize('failure',['short','duplicate_row','wrong_order','duplicate_question','other_question'])
def test_rejects_incomplete_or_wrong_actual_question_coverage(failure):
    ids=[f'q{i}' for i in range(512)]
    rows=[dict(validation_row=i,question_id=ids[i]) for i in range(512)]
    module.verify_rows(rows,ids)
    if failure=='short':rows.pop()
    elif failure=='duplicate_row':rows[-1]['validation_row']=0
    elif failure=='wrong_order':rows[0],rows[1]=rows[1],rows[0]
    elif failure=='duplicate_question':rows[-1]['question_id']=ids[0]
    else:rows[0]['question_id']='another'
    with pytest.raises(RuntimeError):module.verify_rows(rows,ids)
