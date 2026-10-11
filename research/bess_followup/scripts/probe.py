import sys, time
from functools import partial

if __name__ == "__main__":
    import numpy as np
    from easy_marl.examples.bidding.training import make_bess_params, parallel_train
    from easy_marl.examples.bidding.bess_experiment import EVAL_SEED, evaluate_market_metrics
    from easy_marl.src.environment import MARLElectricityMarketEnv

    rounds, steps = int(sys.argv[1]), int(sys.argv[2])
    pf = partial(make_bess_params, power_mw=50, duration_h=4)
    t0 = time.time()
    agents, _ = parallel_train(N=4, num_rounds=rounds, timesteps_per_agent=steps, seed=42,
                               save_dir=None, verbose=False, update_probability=1.0, param_func=pf)
    print(f"TRAIN_SECONDS {time.time() - t0:.1f} rounds={rounds} steps={steps}")
    params = pf(N=4, T=24)
    env = MARLElectricityMarketEnv(agents=agents, params=params, seed=EVAL_SEED, observer_name="simple_v3")
    env.reset(seed=EVAL_SEED)
    print(" h   dem  price  bess_bid   chg   dis    soc   a0     a1")
    for t in range(24):
        a = agents[3].act(env._get_obs(3).copy()); env.step(None, fixed_evaluation=True)
        o = env.output
        print(f"{t:2d} {o['demand'][t]:5.1f} {o['market_prices'][t]:6.1f} {o['bids'][t,3]:8.1f} {o['bess_charge'][t]:5.1f} {o['bess_discharge'][t]:5.1f} {o["soc"][t]:6.1f} {a[0]:6.2f} {a[1]:6.2f}")
    m = evaluate_market_metrics(agents, params, num_episodes=20, seed=EVAL_SEED)
    keys = ["mean_price", "price_std", "daily_spread_mean", "bess_charge_mwh_mean", "bess_discharge_mwh_mean",
            "bess_equivalent_cycles_mean", "bess_mean_charge_price", "bess_mean_discharge_price", "bess_profit_mean"]
    for k in keys:
        print(k, m[k])
    ok = (m["bess_charge_mwh_mean"] > 0 and m["bess_equivalent_cycles_mean"] > 0.3
          and m["bess_mean_charge_price"] is not None and m["bess_mean_discharge_price"] is not None
          and m["bess_mean_charge_price"] < m["bess_mean_discharge_price"])
    print("GATE", "PASS" if ok else "FAIL")
