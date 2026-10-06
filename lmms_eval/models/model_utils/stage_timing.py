"""Per-sample stage timing shared by backends that can hook the vision encoder (``stage_timing=True``)."""

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
