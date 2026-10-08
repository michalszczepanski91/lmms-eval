"""Per-sample timing shared by the HF and vLLM backends: first token, and vision encoder (``stage_timing=True``)."""

import time
from typing import Optional

import torch


class VisionTimer:
    """CUDA events around a vision encoder's forward passes.

    The events are recorded on the current stream from forward pre/post hooks and only read in
    ``pop_seconds``, after the model call has returned, so timing adds no host synchronization to
    the timed path. Several encoder calls between two ``pop_seconds`` calls are summed.
    """

    def __init__(self, visual: torch.nn.Module):
        self._events = []
        visual.register_forward_pre_hook(lambda *_: self._record())
        visual.register_forward_hook(lambda *_: self._record())

    def _record(self):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._events.append(event)

    def pop_seconds(self) -> Optional[float]:
        """Encoder time since the previous call, or None if the encoder did not run."""
        events, self._events = self._events, []
        if len(events) < 2:
            return None
        events[-1].synchronize()
        return sum(start.elapsed_time(end) for start, end in zip(events[::2], events[1::2])) / 1000


class FirstTokenTimer:
    """Generation streamer (``generate(streamer=...)``) that records when the first new token is produced.

    ``generate`` calls ``put`` once with the prompt ids, then once per decoding step with ids already copied to
    CPU, so the second call marks the first token. HF streamers support batch size 1 only.
    """

    def __init__(self):
        self._calls = 0
        self.first_token_time = None

    def put(self, value):
        self._calls += 1
        if self._calls == 2:
            self.first_token_time = time.perf_counter()

    def end(self):
        pass
