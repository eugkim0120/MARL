import numpy as np
from pathlib import Path
from easy_marl.examples.bidding import bess_experiment as E
print("seed meanP | hrs/day P<=45 | hrs/day 45<P<75 | hrs/day P>=75 | hrs where charging pays vs best sell (P*0.9 < sellP-?)")
for seed in range(42, 52):
    gens = E.get_pretrained(seed, E.PRESETS["full"], Path("outputs/bess_sweep/full"))
    P = E.simulate_episodes(gens, E.make_default_params(N=3, T=24, day_ahead=True), 20, E.EVAL_SEED)["market_prices"]
    lo, hi = (P <= 45).sum(1).mean(), (P >= 75).sum(1).mean()
    print(f"{seed}  {P.mean():5.1f} | {lo:4.1f} | {24-lo-hi:4.1f} | {hi:4.1f}")
