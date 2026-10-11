"""
Battery (BESS) sweep: 3 generators plus one PPO-trained battery, varying battery
power and duration, compared against the same market with no battery.

Design: for each seed the three generators are trained alone first (pretraining) and
kept. The no-battery baseline is those generators on their own, and every battery
config adds a battery next to the very same generators, so effects are measured as
paired per-seed differences. Two arms:
    frozen:   generators stay fixed, only the battery learns (short-run effect)
    adaptive: everyone keeps training (long-run effect); its baseline is the
              pretrained generators trained for the same extra rounds without a battery

Usage:
    python -m easy_marl.examples.bidding.bess_experiment run --preset smoke
    python -m easy_marl.examples.bidding.bess_experiment aggregate --out outputs/bess_sweep/smoke
"""

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from easy_marl.src.agents import PPOAgent
from easy_marl.src.environment import MARLElectricityMarketEnv
from easy_marl.examples.bidding.training import (
    DEFAULT_OBS,
    init_agents,
    make_bess_params,
    make_default_params,
    parallel_train,
)

N_GENERATORS = 3
HOURS = 24
ARMS = ("frozen", "adaptive")
# Same demand draws for every config, so differences come from the battery, not noise.
EVAL_SEED = 10_000
# A battery below this many equivalent cycles has only sold its free starting charge.
MIN_USEFUL_CYCLES = 0.6

# Day-ahead training: one environment step is one simulated day, so
# timesteps_per_agent counts days.
PRESETS = {
    "smoke": {
        "powers": [10, 25, 50],
        "durations": [1, 4],
        "seeds": list(range(42, 52)),
        "arms": ["frozen"],
        "pretrain_rounds": 8,
        "pretrain_timesteps_per_agent": 5_000,
        "pretrain_change_tol": 0.05,
        "num_rounds": 3,
        "timesteps_per_agent": 5_000,
        "eval_episodes": 20,
        "update_probability": 1.0,
    },
    "full": {
        "powers": [5, 10, 25, 50, 75],
        "durations": [1, 2, 4, 8],
        "seeds": list(range(42, 52)),
        "arms": ["frozen"],
        "pretrain_rounds": 10,
        "pretrain_timesteps_per_agent": 20_000,
        "pretrain_change_tol": 0.05,
        "num_rounds": 5,
        "timesteps_per_agent": 20_000,
        "eval_episodes": 20,
        "update_probability": 1.0,
    },
}


@dataclass(frozen=True)
class SweepConfig:
    power_mw: Optional[float]
    duration_h: Optional[float]
    seed: int
    arm: str = "frozen"

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f"Unknown arm {self.arm!r}, expected one of {ARMS}.")

    @property
    def is_baseline(self) -> bool:
        return self.power_mw is None

    @property
    def config_id(self) -> str:
        if self.is_baseline:
            base = f"baseline_s{self.seed}"
        else:
            base = f"p{self.power_mw:g}_d{self.duration_h:g}_s{self.seed}"
        return base if self.arm == "frozen" else f"{base}_{self.arm}"

    @property
    def n_agents(self) -> int:
        return N_GENERATORS if self.is_baseline else N_GENERATORS + 1

    def param_func(self):
        # Every agent, baseline generators included, commits to a full day at once.
        if self.is_baseline:
            return partial(make_default_params, day_ahead=True)
        return partial(
            make_bess_params,
            power_mw=self.power_mw,
            duration_h=self.duration_h,
            day_ahead=True,
        )


def build_configs(powers, durations, seeds, arms=("frozen",)) -> List[SweepConfig]:
    configs = []
    for seed in seeds:
        for arm in arms:
            configs.append(SweepConfig(None, None, seed, arm))
            for power in powers:
                for duration in durations:
                    configs.append(SweepConfig(power, duration, seed, arm))
    return configs


