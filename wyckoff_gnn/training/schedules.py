"""Learning rate schedules for WyckoffGNN training.

PolynomialDecaySchedule matches coGN's kgcnn KerasPolynomialDecaySchedule:
    lr(step) = (lr_start - lr_stop) * (1 - step/decay_steps)^power + lr_stop
Updates per optimizer step (not per epoch).
"""

from __future__ import annotations

import math


class PolynomialDecaySchedule:
    """Polynomial learning rate schedule matching TF's KerasPolynomialDecaySchedule.

    lr(step) = (lr_start - lr_stop) * (1 - step/decay_steps)^power + lr_stop
    with power=1.0 (linear decay) by default.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        epochs: int,
        lr_start: float = 1e-3,
        lr_stop: float = 1e-5,
        power: float = 1.0,
    ):
        self.lr_start = float(lr_start)
        self.lr_stop = float(lr_stop)
        self.power = float(power)
        steps_per_epoch = dataset_size / batch_size
        self.decay_steps = int(math.ceil(epochs * steps_per_epoch))
        self._step = 0

    def step(self):
        self._step += 1

    def get_lr(self) -> float:
        s = min(self._step, self.decay_steps)
        return (self.lr_start - self.lr_stop) * (
            (1.0 - s / self.decay_steps) ** self.power
        ) + self.lr_stop

    def state_dict(self):
        return {
            "lr_start": self.lr_start,
            "lr_stop": self.lr_stop,
            "power": self.power,
            "decay_steps": self.decay_steps,
            "_step": self._step,
        }

    @classmethod
    def from_state_dict(cls, d):
        s = cls.__new__(cls)
        s.lr_start = d["lr_start"]
        s.lr_stop = d["lr_stop"]
        s.power = d["power"]
        s.decay_steps = d["decay_steps"]
        s._step = d.get("_step", 0)
        return s
