"""Pure monitor control, CPU-fixturable without actual Popen/GPU/sleep."""

def supervise(children,observe,renew,progress,sleep,spawn_error=None,on_failure=None):
    failure=spawn_error
    revoked=False
    while any(x['process'].poll() is None for x in children):
        if failure is not None and not revoked and on_failure:
            on_failure(failure);revoked=True
        if failure is None:
            try:
                observe()
                for x in children:
                    if x['process'].poll() is None:renew(x)
            except BaseException as error:
                failure=repr(error)
                if on_failure:on_failure(failure);revoked=True
        progress(failure)
        sleep(10)
    for x in children:x['process'].wait()
    return failure
