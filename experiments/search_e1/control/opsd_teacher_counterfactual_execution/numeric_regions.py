"""Explicit teacher-presence overlay; reference success remains an honest label.

Only frozen probe's has_teacher predicate is AST-overlaid. All numeric/mask/clip/
scalar/invariance checks stay unchanged. Frozen module/source never modified.
"""
import ast
import inspect


def probe(region,summary,token_path,response,spans,teacher_present):
    source=inspect.getsource(region.probe);tree=ast.parse(source);changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='has_teacher' for x in node.targets):
            expected=ast.parse("summary['reference_status'] == 'successful_original_reference'",mode='eval').body
            if ast.dump(node.value)!=ast.dump(expected):raise ValueError('Frozen region predicate changed')
            node.value=ast.parse("summary['_M3_teacher_present_for_math'] is True",mode='eval').body;changed+=1
    if changed!=1:raise ValueError('Expected exactly one frozen reference classifier')
    namespace=region.__dict__.copy();exec(compile(ast.fix_missing_locations(tree),region.__file__+':M3-teacher-presence-overlay','exec'),namespace)
    value=dict(summary,_M3_teacher_present_for_math=bool(teacher_present))
    result=namespace['probe'](value,token_path,response,spans)
    result['M3_math_overlay']=dict(teacher_present=bool(teacher_present),reference_status_unchanged=summary['reference_status'],
                                  frozen_source=region.__file__,changed_predicate_only=True,success_inferred=False)
    return result
