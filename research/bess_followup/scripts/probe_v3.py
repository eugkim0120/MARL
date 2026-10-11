import json, sys, time
from pathlib import Path

if __name__ == "__main__":
    from easy_marl.examples.bidding.bess_experiment import (
        EVAL_SEED, PRESETS, SweepConfig, get_pretrained, run_config, HOURS,
    )
    from easy_marl.examples.bidding.training import make_bess_params, init_agents
    from easy_marl.src.environment import MARLElectricityMarketEnv

    preset = dict(PRESETS["smoke"])
    for kv in sys.argv[2:]:
        k, v = kv.split("=")
        preset[k] = type(preset[k])(v) if not isinstance(preset[k], float) else float(v)
    out = Path(sys.argv[1])
    t0 = time.time()
    get_pretrained(42, preset, out)
    print(f"PRETRAIN_SECONDS {time.time() - t0:.1f}", flush=True)
    conv = json.loads((out / "pretrain/s42/convergence.json").read_text())
    print("policy_change_per_round", [round(c, 4) for c in conv["policy_change_per_round"]])
    cfg = SweepConfig(power_mw=50, duration_h=4, seed=42)
    t0 = time.time()
    res = run_config(cfg, preset, out)
    print(f"BATTERY_CONFIG_SECONDS {time.time() - t0:.1f}", flush=True)
    m = res["metrics"]
    for k in ["mean_price", "price_std", "daily_spread_mean", "bess_charge_mwh_mean",
              "bess_discharge_mwh_mean", "bess_equivalent_cycles_mean",
              "bess_mean_charge_price", "bess_mean_discharge_price", "bess_profit_mean"]:
        print(k, m[k])
    base = run_config(SweepConfig(None, None, 42), preset, out)["metrics"]
    print("baseline mean_price", base["mean_price"], "price_std", base["price_std"])
    ok = (m["bess_charge_mwh_mean"] > 0 and m["bess_equivalent_cycles_mean"] > 0.3
          and m["bess_mean_charge_price"] is not None and m["bess_mean_discharge_price"] is not None
          and m["bess_mean_charge_price"] < m["bess_mean_discharge_price"])
    print("GATE", "PASS" if ok else "FAIL")
