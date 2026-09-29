import signal
from contextlib import contextmanager


class TimeoutException(BaseException):
    pass


@contextmanager
def time_limit(t):
    if t < 0:
        yield
    else:
        def signal_handler(signum, frame):
            raise TimeoutException(f"Operation exceeded {t} seconds")

        signal.signal(signal.SIGALRM, signal_handler)
        signal.setitimer(signal.ITIMER_REAL, t)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
