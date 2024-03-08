from contextlib import contextmanager
import threading
import _thread


def timeout(is_finished: threading.Event):
    is_finished.set()
    _thread.interrupt_main()


@contextmanager
def time_limit(seconds):
    if seconds is None:
        yield
        return
    is_finished = threading.Event()
    timer = threading.Timer(seconds, lambda: timeout(is_finished))
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        if not is_finished.is_set():
            raise KeyboardInterrupt("Job cancelled!")
        else:
            raise TimeoutError("Timed out!")
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()

