"""Action-boundary batching of independent question planners.

One unresolved continuation per question. A finished question can select its
next branch while other questions continue older branches. Native generation,
masking, reward and Planner selection/backoff/fill rules remain unchanged.
Shared RNG consumption order and generation batch shapes can differ; this is
not a promise of bitwise reproduction of the whole-suffix scheduler.
"""
import copy
import time


def run_frontier(planners, engine, meta, finish_child):
    from .tree_audit import state_signature
    inflight={}
    try:
        while True:
            for question,planner in enumerate(planners):
                if question in inflight:continue
                request=planner.next_request()
                if request is None:continue
                parent,index,attempt=request
                snapshot=parent.snapshots[index];child=snapshot.branch()
                if state_signature(snapshot)!=state_signature(child):
                    raise ValueError('Branch changed executor state or inherited probabilities')
                if not engine.pending(child):
                    raise ValueError('Restored branch has no pending native action')
                inflight[question]=dict(planner=planner,request=request,child=child,
                    suffix=[],started=time.time())
            if not inflight:break
            # Question order controls transport only. No question gets two
            # concurrent branches, and no unavailable parent reward is guessed.
            active=[inflight[q] for q in sorted(inflight)]
            for task in active:task['suffix'].append(copy.deepcopy(task['child']))
            engine.advance([task['child'] for task in active],meta)
            finished=[q for q in sorted(inflight) if not engine.pending(inflight[q]['child'])]
            children=[inflight[q]['child'] for q in finished]
            rewards=engine.reward_callback(children) if children else []
            if len(rewards)!=len(children):raise ValueError('Finished frontier reward rows differ')
            for question,reward in zip(finished,rewards):
                task=inflight[question]
                finish_child(task['planner'],task['request'],task['child'],
                             task['suffix'],reward,task['started'])
                engine.cost['trajectories']+=1
                del inflight[question]
            engine.timing('recursive_frontier_action_batch',active_questions=len(active),
                completed_questions=len(finished),still_pending_questions=len(inflight),
                next_ready_questions=len(finished),scope='one_pending_branch_per_question')
    except Exception as error:
        for planner in planners:
            if planner.pending is not None:planner.accept_result(error=error)
        for planner in planners:
            if planner.reason is None:
                planner.reason='batch_execution_aborted';planner.end_search(planner.reason)
            planner.finish()
        raise
