"""Adapt actual V2 audit without erasing ownership observations."""
import inspect


def strict_scan(gates,run_id,uid,controller,known,roots,exclusions=(),expected_live_owner=None,live=None):
    parameters=inspect.signature(gates.owned_run_processes).parameters
    if not {'known_owned','roots','ignored_exact','ancestry_anchors'}.issubset(parameters):
        raise ValueError('Actual V2 strict process scanner required; no env-only fallback')
    audit=gates.owned_run_processes(run_id,uid,controller,known,roots,exclusions,
                                  ignored_exact=[expected_live_owner] if expected_live_owner else [],
                                  ancestry_anchors=[controller])
    return classify(audit,expected_live_owner,live(expected_live_owner['pid']) if expected_live_owner else None)


def classify(audit,owner=None,actual=None):
    if audit.get('schema')!='same-uid-owned-process-audit-v2':raise ValueError('Unknown process audit schema')
    expected=[];unexpected=[]
    valid_owner=(owner is not None and actual is not None and all(owner[k]==actual[k] for k in ('pid','uid','starttick')) and actual.get('state')!='Z')
    for entry in audit['blockers']:
        is_owner=valid_owner and all(entry.get(k)==owner[k] for k in ('pid','uid','starttick'))
        if is_owner and entry.get('state')!='Z' and entry['reason']=='known_owned_or_run_or_argv_or_descendant_present':
            expected.append(dict(entry,classification='expected_live_M3_controller_not_claimed_exited'))
        else:unexpected.append(entry)
    return dict(raw_audit=audit,expected_still_running_controller=expected,
                actual_expected_controller=actual,unexpected_blockers=unexpected,
                accepted_no_unexpected_owned_or_opaque=not unexpected,
                all_owned_absent_claimed=False if expected else audit['accepted_no_owned_or_opaque'])
