import time
from contextlib import contextmanager

class ProgressLogger:
    """Small stage-level logger that works in CLI and notebooks."""
    def __init__(self, enabled=True, prefix="TS Advisor"):
        self.enabled = enabled
        self.prefix = prefix
        self.start = time.time()

    def stage(self, title):
        if self.enabled:
            print(f"\n[{self.prefix}] {title}")
            print("-" * (len(title) + len(self.prefix) + 3))

    def log(self, message):
        if self.enabled:
            print(f"[{time.time() - self.start:6.1f}s] {message}", flush=True)

    @contextmanager
    def timed(self, message):
        self.log(message)
        t0 = time.time()
        yield
        self.log(f"Finished {message} ({time.time() - t0:.1f}s)")
