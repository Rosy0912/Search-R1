"""Execute fixed original main statement blocks, never rewritten reward math."""
import ast, hashlib, pathlib, types
DUMP_SHA='816052e763ff52f4322932372a58ba0b438710af50db5f21a82631241f246719'
TRACE_KEYS=('rows','prompts','golds_all','questions','nq','layers','cb_prompt','cb_ans','q_cb','flat_prompts','flat_qi','gens','valid','commit_ctx','ans_ctx','flip_ctx','flip2_ctx','answers','ems','flip_ans','flips','flips2')

def source(path):
    raw=pathlib.Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=DUMP_SHA:raise ValueError('original dump bytes changed')
    return ast.parse(raw)

def target(node,name):
    return isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id==name for x in node.targets)

def blocks(path):
    body=next(n for n in source(path).body if isinstance(n,ast.FunctionDef) and n.name=='main').body
    start=lambda name:next(i for i,n in enumerate(body) if target(n,name))
    prep=start('df');generation=start('cb_prompt');hidden=start('Hc')
    assembly=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple) and any(isinstance(x,ast.Name) and x.id=='y_em' for x in n.targets[0].elts))
    before_llm=start('llm')
    return {'prepare':body[prep:before_llm],'generation':body[generation:hidden],'hidden':body[hidden:assembly],'assembly':body[assembly:]}

def execute(path,block,namespace):
    nodes=blocks(path)[block]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path)+':'+block,'exec'),namespace)
    return namespace

def block_hashes(path):return {k:hashlib.sha256(ast.dump(ast.Module(body=v,type_ignores=[])).encode()).hexdigest() for k,v in blocks(path).items()}

def trace(namespace):
    value={key:namespace[key] for key in TRACE_KEYS}
    # NumPy scalar labels become JSON scalar values; no token/text transformation.
    def native(v):
        if isinstance(v,dict):return {str(k):native(x) for k,x in v.items()}
        if isinstance(v,(tuple,list)):return [native(x) for x in v]
        if hasattr(v,'item'):return v.item()
        return v
    return native(value)

def restore(value,args,namespace):
    if set(value)!=set(TRACE_KEYS):raise ValueError('incomplete/unknown generation trace')
    namespace.update(value);namespace['args']=args
    if len(value['gens'])!=value['nq']*args.G:raise ValueError('rollout row count')
    if len(value['cb_ans'])!=value['nq'] or len(value['q_cb'])!=value['nq']:raise ValueError('closed-book row count')
    n=len(value['valid'])
    if value['valid']!=sorted(set(value['valid'])) or any(not 0<=x<len(value['gens']) for x in value['valid']):raise ValueError('valid row order')
    for key in ('commit_ctx','ans_ctx','flip_ctx','flip2_ctx','answers','ems','flip_ans','flips','flips2'):
        if len(value[key])!=n:raise ValueError('generation trace alignment: '+key)
    for key in ('q_cb','ems','flips','flips2'):
        if any(x not in (0,1) for x in value[key]):raise ValueError('nonbinary original label')
    return namespace
