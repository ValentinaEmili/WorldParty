import sys, os, runpy, logging

try:
    import absl.logging  # import first so absl creates its own logger class
except Exception:
    pass

class _Drop(logging.Filter):
    def filter(self, record):
        return "Received unexpected argument" not in record.getMessage()

logging.getLogger("absl").addFilter(_Drop())

try:
    import torch
    torch.backends.nnpack.set_flags(False)
except Exception:
    pass

script = os.path.abspath(sys.argv[1])
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(script))  # mimic `python path/to/script.py`
runpy.run_path(script, run_name="__main__")
