from time import time
from functools import wraps

def timetrackerLog(message):
    print('[TimeTracker] {function_name} {total_time:.3f}'.format(**message))

def time_tracker():
    def _simple_time_tracker(fn):
        @wraps(fn)
        def wrapped_fn(*args, **kwargs):
            start_time = time()

            try:
                result = fn(*args, **kwargs)
            finally:
                elapsed_time = time() - start_time

                # log the result
                timetrackerLog({
                    'function_name': fn.__name__,
                    'total_time': elapsed_time,
                })
                
            return result

        return wrapped_fn
    return _simple_time_tracker