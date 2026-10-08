#!/usr/bin/env python3
"""Separate explicit arithmetic from raw JSON, not the first audit's output."""
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from pathlib import Path

W = Path(__file__).resolve().parent
P = W / 'actual-snapshots/3261-20261006T030139996224Z/original-files/ray-stdout-pid-1533312.out'
raw = P.read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'f4b307b89fa0081cf21be934bddbba0b418fb64d75e1df488a369a7010a26300'
events = [json.loads(line.split(b'=', 1)[1], parse_float=Decimal) for line in raw.split(b'\n')
          if line.startswith(b'BASELINE_EVALUATION_METRICS_JSON=')]
assert len(events) == 8
schedule = [('dev',0,256),('stage',5,32),('dev',50,256),('stage',50,32),
            ('dev',100,256),('stage',100,32),('dev',150,256),('stage',150,32)]
expected_em_source_counts = [(51,43),(2,1),(9,22),(2,5),(11,12),(2,1),(14,17),(1,2)]
expected_full_format_source_counts = [(162,105),(0,0),(1,0),(0,1),(0,0),(0,0),(0,0),(0,0)]
table = []
for index, (kind,step,n) in enumerate(schedule):
    m=events[index]
    correct=tuple(int(Fraction(m['val/em/'+s])*n) for s in ('nq','hotpotqa'))
    formatting=tuple(int(Fraction(m['val/format_rate/'+s])*n) for s in ('nq','hotpotqa'))
    assert correct == expected_em_source_counts[index]
    assert formatting == expected_full_format_source_counts[index]
    assert Fraction(m['val/em/all']) == Fraction(sum(correct),2*n)
    assert m['val/test_score/nq'] == m['val/em/nq'] and m['val/test_score/hotpotqa'] == m['val/em/hotpotqa']
    f1=(m['val/f1/nq']*n+m['val/f1/hotpotqa']*n)/(2*n)
    assert abs(f1-m['val/f1/all']) < Decimal('1e-15')
    tokens=tuple(int(Fraction(m['val/generated_tokens/'+s])*n) for s in ('nq','hotpotqa'))
    searches=tuple(int(Fraction(m['val/search_calls/'+s])*n) for s in ('nq','hotpotqa'))
    assert Fraction(m['val/generated_tokens/all']) == Fraction(sum(tokens),2*n)
    assert Fraction(m['val/search_calls/all']) == Fraction(sum(searches),2*n)
    table.append({'kind':kind,'step':step,'nq_correct':correct[0],'hotpot_correct':correct[1],
                  'all_correct':sum(correct),'count':2*n,'EM':str(m['val/em/all']),
                  'F1':str(m['val/f1/all']),'format_correct':sum(formatting),
                  'token_counts':tokens,'token_total':sum(tokens),'search_counts':searches,'search_total':sum(searches)})

# Explicit hand calculations for the newly added150 pair.
assert Fraction(14,256)+Fraction(17,256) == 2*Fraction(31,512)
assert Fraction(1,32)+Fraction(2,32) == 2*Fraction(3,64)
assert (Decimal('0.08881265242818273')+Decimal('0.08116585250929743'))/2 == Decimal('0.08498925246874008')
assert (Fraction(1,24)+Fraction(1,16))/2 == Fraction(5,96)
assert table[6]['token_counts'] == (152177,183130) and table[6]['token_total'] == 335307
assert table[6]['search_counts'] == (148,153) and table[6]['search_total'] == 301
assert table[7]['token_counts'] == (17602,20986) and table[7]['token_total'] == 38588
assert table[7]['search_counts'] == (18,15) and table[7]['search_total'] == 33
print(json.dumps({'status':'PASS_SEPARATE_RAW_DECIMAL_FRACTION_AND_LITERAL150_MATH',
                  'raw_driver_sha256':hashlib.sha256(raw).hexdigest(),'events':table,
                  '150_dev_EM_drop_vs_initial_exact_fraction':str(Fraction(31-94,512)),
                  'scope':'Arithmetic on saved source aggregates, not per-question rescore or significance.'},indent=2))
