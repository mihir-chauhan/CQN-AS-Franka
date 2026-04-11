import time
import torch

class Timer(object):

    def __init__(self):
        self._time = None

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._time = time.time() - self._start_time

    def __call__(self):
        return self._time

def metrics_full_log(metrics: dict, key: str, value: torch.Tensor):
    metrics[f"{key}_mean"] = value.mean().item()
    metrics[f"{key}_std"] = value.std().item()
    metrics[f"{key}_min"] = value.min().item()
    metrics[f"{key}_max"] = value.max().item()

    return metrics
