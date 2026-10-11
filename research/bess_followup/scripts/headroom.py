import numpy as np
from pathlib import Path
from easy_marl.examples.bidding import bess_experiment as E
cfg = E.SweepConfig(75, 4, 43, "frozen")
p = cfg.param_func()(N=4, T=24)
print({k: (v if not isinstance(v, (list, np.ndarray)) else np.round(v, 1).tolist()) for k, v in p.items() if k in ("demand_profile","capacities","K","costs","c","demand_scale")})
