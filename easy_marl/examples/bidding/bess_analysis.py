"""
Numbers behind the battery (BESS) sweep dashboard: paired comparisons of each battery
config against the same-seed no-battery market, built from metrics.json and dispatch.npz.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from easy_marl.examples.bidding.bess_experiment import (
    DELTA_METRICS,
    bootstrap_ci,
    load_results,
)


def paired_rows(rows: List[Dict], arm: str) -> List[Tuple[Dict, Dict]]:
    """Every battery row of an arm with the baseline row of the same seed and arm."""
    baselines = {r["seed"]: r for r in rows if r["arm"] == arm and r["power_mw"] is None}
    pairs = []
    for r in rows:
        if r["arm"] != arm or r["power_mw"] is None:
            continue
        if r["seed"] not in baselines:
            raise ValueError(
                f"No baseline for {r['config_id']} (arm {arm!r}, seed {r['seed']})."
            )
        pairs.append((r, baselines[r["seed"]]))
    return pairs


def _hourly_prices(out_dir: Path) -> Dict[str, np.ndarray]:
    return {
        path.parent.name: np.array(
            json.loads(path.read_text())["metrics"]["mean_hourly_price"], dtype=np.float64
        )
        for path in Path(out_dir).glob("*/metrics.json")
    }


def hourly_deltas(out_dir, arm: str) -> Dict[Tuple[float, float], List[float]]:
    """Per (power, duration): mean over seeds of battery hourly price minus the same-seed baseline."""
    rows = load_results(out_dir)
    hourly = _hourly_prices(Path(out_dir))
    per_config: Dict[Tuple[float, float], List[np.ndarray]] = {}
    for r, base in paired_rows(rows, arm):
        delta = hourly[r["config_id"]] - hourly[base["config_id"]]
        per_config.setdefault((r["power_mw"], r["duration_h"]), []).append(delta)
    return {key: np.mean(deltas, axis=0).tolist() for key, deltas in sorted(per_config.items())}


def baseline_hourly(out_dir, arm: str) -> np.ndarray:
    rows = load_results(out_dir)
    hourly = _hourly_prices(Path(out_dir))
    profiles = [hourly[r["config_id"]] for r in rows if r["arm"] == arm and r["power_mw"] is None]
    if not profiles:
        raise ValueError(f"No baseline runs for arm {arm!r}.")
    return np.mean(profiles, axis=0)


def load_dispatch(out_dir, config_id: str) -> Dict[str, np.ndarray]:
    path = Path(out_dir) / config_id / "dispatch.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"No dispatch for {config_id} ({path}). Run "
            f"`python -m easy_marl.examples.bidding.bess_experiment replay --out {out_dir}` "
            "to regenerate it from the saved agents."
        )
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _hourly_dispatch(dispatch: Dict[str, np.ndarray]) -> Dict[str, Optional[np.ndarray]]:
    """Episode-averaged hourly generator output, battery flows and price of one config."""
    n_generators = len(dispatch["generator_cost"])
    has_battery = "bess_discharge" in dispatch
    return {
        "generators": dispatch["q_cleared"][:, :, :n_generators].mean(axis=0).T,
        "discharge": dispatch["bess_discharge"].mean(axis=0) if has_battery else None,
        "charge": dispatch["bess_charge"].mean(axis=0) if has_battery else None,
        "price": dispatch["market_prices"].mean(axis=0),
        "demand": dispatch["demand"].mean(axis=0),
    }


def _mean_over_seeds(items: List[Dict[str, Optional[np.ndarray]]]) -> Dict[str, Optional[np.ndarray]]:
    return {
        key: None if items[0][key] is None else np.mean([item[key] for item in items], axis=0)
        for key in items[0]
    }


def dispatch_profile(out_dir, arm: str, power: float, duration: float) -> Dict:
    """Seed-averaged hourly dispatch of one battery config next to its paired baselines."""
    rows = load_results(out_dir)
    pairs = [
        (r, base)
        for r, base in paired_rows(rows, arm)
        if r["power_mw"] == power and r["duration_h"] == duration
    ]
    if not pairs:
        raise ValueError(f"No config {power:g} MW / {duration:g} h in arm {arm!r}.")
    battery = _mean_over_seeds(
        [_hourly_dispatch(load_dispatch(out_dir, r["config_id"])) for r, _ in pairs]
    )
    baseline = _mean_over_seeds(
        [_hourly_dispatch(load_dispatch(out_dir, base["config_id"])) for _, base in pairs]
    )
    return {"demand": battery.pop("demand"), "battery": battery, "baseline": baseline}


def _plant_output(out_dir, config_id: str, cache: Dict) -> np.ndarray:
    """Hourly output per plant: one row per generator, then the battery's net output (discharge minus charge)."""
    if config_id not in cache:
        h = _hourly_dispatch(load_dispatch(out_dir, config_id))
        battery = np.zeros(h["generators"].shape[1]) if h["discharge"] is None else h["discharge"] - h["charge"]
        cache[config_id] = np.vstack([h["generators"], battery])
    return cache[config_id]


def generation_by_hour(out_dir, arm: str) -> Dict:
    """Seed-mean hourly output per plant: a "baseline" entry and one per (power, duration)."""
    rows = load_results(out_dir)
    cache: Dict[str, np.ndarray] = {}
    per_config: Dict[Tuple[float, float], List[np.ndarray]] = {}
    baselines: Dict[int, np.ndarray] = {}
    for r, base in paired_rows(rows, arm):
        per_config.setdefault((r["power_mw"], r["duration_h"]), []).append(
            _plant_output(out_dir, r["config_id"], cache)
        )
        baselines[base["seed"]] = _plant_output(out_dir, base["config_id"], cache)
    levels = {key: np.mean(items, axis=0) for key, items in sorted(per_config.items())}
    return {"baseline": np.mean(list(baselines.values()), axis=0), **levels}


def generation_change_by_hour(out_dir, arm: str) -> Dict[Tuple[float, float], np.ndarray]:
    """Per (power, duration): seed-mean hourly output change per plant against the same-seed baseline."""
    rows = load_results(out_dir)
    cache: Dict[str, np.ndarray] = {}
    per_config: Dict[Tuple[float, float], List[np.ndarray]] = {}
    for r, base in paired_rows(rows, arm):
        delta = _plant_output(out_dir, r["config_id"], cache) - _plant_output(out_dir, base["config_id"], cache)
        per_config.setdefault((r["power_mw"], r["duration_h"]), []).append(delta)
    return {key: np.mean(deltas, axis=0) for key, deltas in sorted(per_config.items())}


def effect_grid(effects: List[Dict], arm: str, metric: str):
    """Mean paired difference and significance on the power x duration grid."""
    if metric not in DELTA_METRICS:
        raise ValueError(f"Unknown metric {metric!r}, expected one of {DELTA_METRICS}.")
    cells = {
        (e["power_mw"], e["duration_h"]): e
        for e in effects
        if e["arm"] == arm and e["metric"] == metric
    }
    powers = sorted({p for p, _ in cells})
    durations = sorted({d for _, d in cells})
    means = np.full((len(powers), len(durations)), np.nan)
    significant = np.zeros((len(powers), len(durations)), dtype=bool)
    for (power, duration), e in cells.items():
        i, j = powers.index(power), durations.index(duration)
        means[i, j] = e["mean_delta"]
        significant[i, j] = e["significant"]
    return powers, durations, means, significant


PDC_POINTS = 101
SCARCITY_PERCENTILE = 95


def _peak_hours(demand: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Highest- and lowest-demand hours of every episode (about a sixth of the day each)."""
    k = max(1, demand.shape[1] // 6)
    order = np.argsort(demand, axis=1, kind="stable")
    return order[:, -k:], order[:, :k]


def _hours_mean(values: np.ndarray, hours: np.ndarray) -> float:
    return float(np.take_along_axis(values, hours, axis=1).mean())


def pair_metrics(base: Dict[str, np.ndarray], battery: Dict[str, np.ndarray], power_mw: float, duration_h: float) -> Dict[str, np.ndarray]:
    """What the battery changed for one seed, from the two recorded dispatches (same demand episodes)."""
    n_gen = len(base["generator_cost"])
    cost = base["generator_cost"].astype(np.float64)
    demand = base["demand"].astype(np.float64)
    price_b = base["market_prices"].astype(np.float64)
    price_x = battery["market_prices"].astype(np.float64)
    q_b = base["q_cleared"][:, :, :n_gen].astype(np.float64)
    q_x = battery["q_cleared"][:, :, :n_gen].astype(np.float64)
    charge = battery["bess_charge"].astype(np.float64)
    discharge = battery["bess_discharge"].astype(np.float64)
    net = discharge - charge
    energy = power_mw * duration_h

    gen_profit_b = ((price_b[:, :, None] - cost) * q_b).sum(axis=1)
    gen_profit_x = ((price_x[:, :, None] - cost) * q_x).sum(axis=1)
    delta_cost = ((q_x - q_b) * cost).sum(axis=(1, 2)).mean()
    consumer_b = (price_b * demand).sum(axis=1)
    consumer_x = (price_x * demand).sum(axis=1)
    daily_profit = (price_x * net).sum(axis=1)
    profit_at_baseline = float((price_b * net).sum(axis=1).mean())
    peak, off_peak = _peak_hours(demand)
    threshold = np.percentile(price_b, SCARCITY_PERCENTILE)
    quantiles = np.linspace(0, 100, PDC_POINTS)

    delta_consumer = float((consumer_x - consumer_b).mean())
    delta_gen_profit = float((gen_profit_x - gen_profit_b).sum(axis=1).mean())
    return {
        "baseline_consumer_cost": float(consumer_b.mean()),
        "battery_profit": float(daily_profit.mean()),
        "battery_profit_at_baseline_prices": profit_at_baseline,
        "cannibalisation": profit_at_baseline - float(daily_profit.mean()),
        "daily_profit": daily_profit,
        "loss_day_share": float((daily_profit < 0).mean()),
        "delta_consumer_cost": delta_consumer,
        "delta_generator_profit": delta_gen_profit,
        "delta_generator_profit_by_plant": (gen_profit_x - gen_profit_b).mean(axis=0),
        "delta_generation_cost": float(delta_cost),
        "welfare_residual": delta_consumer - delta_gen_profit - float(delta_cost) - float(daily_profit.mean()),
        "delta_peak_price": _hours_mean(price_x, peak) - _hours_mean(price_b, peak),
        "delta_offpeak_price": _hours_mean(price_x, off_peak) - _hours_mean(price_b, off_peak),
        "delta_scarcity_pp": 100.0 * float((price_x >= threshold).mean() - (price_b >= threshold).mean()),
        "peak_shaving_mw": float(demand.max(axis=1).mean() - (demand - net).max(axis=1).mean()),
        "utilisation": float(((charge + discharge) > 1e-6).mean()),
        "net_frac_by_hour": net.mean(axis=0) / power_mw,
        "soc_frac_by_hour": battery["soc"].astype(np.float64).mean(axis=0) / energy,
        "pdc_base": np.percentile(price_b, quantiles),
        "pdc_battery": np.percentile(price_x, quantiles),
    }


def config_metrics(out_dir, arm: str) -> Dict[Tuple[float, float], Dict[str, np.ndarray]]:
    """Per (power, duration): every pair_metrics entry stacked over seeds (first axis, sorted by seed)."""
    rows = load_results(out_dir)
    cache: Dict[str, Dict[str, np.ndarray]] = {}

    def dispatch(config_id: str):
        if config_id not in cache:
            cache[config_id] = load_dispatch(out_dir, config_id)
        return cache[config_id]

    per_config: Dict[Tuple[float, float], List[Tuple[int, Dict]]] = {}
    for r, base in paired_rows(rows, arm):
        metrics = pair_metrics(dispatch(base["config_id"]), dispatch(r["config_id"]), r["power_mw"], r["duration_h"])
        per_config.setdefault((r["power_mw"], r["duration_h"]), []).append((r["seed"], metrics))
    stacked = {}
    for key, items in sorted(per_config.items()):
        items.sort(key=lambda item: item[0])
        stacked[key] = {
            name: np.array([m[name] for _, m in items]) for name in items[0][1]
        }
    return stacked


def _mean_with_significance(values: np.ndarray) -> Tuple[float, bool]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float(values.mean()), False
    low, high = bootstrap_ci(values)
    return float(values.mean()), bool(low > 0 or high < 0)


def metric_grid(metrics: Dict[Tuple[float, float], Dict[str, np.ndarray]], name: str):
    """Seed-mean of a config_metrics scalar and its significance on the power x duration grid."""
    sample = next(iter(metrics.values()))
    if name not in sample or sample[name].ndim != 1:
        raise ValueError(f"Unknown metric {name!r}, expected a per-seed scalar from config_metrics.")
    powers = sorted({p for p, _ in metrics})
    durations = sorted({d for _, d in metrics})
    means = np.full((len(powers), len(durations)), np.nan)
    significant = np.zeros((len(powers), len(durations)), dtype=bool)
    for (power, duration), values in metrics.items():
        i, j = powers.index(power), durations.index(duration)
        means[i, j], significant[i, j] = _mean_with_significance(values[name])
    return powers, durations, means, significant


def _price_effect(effects: List[Dict], arm: str, power: float, duration: float) -> Dict:
    return next(
        e
        for e in effects
        if (e["arm"], e["power_mw"], e["duration_h"], e["metric"]) == (arm, power, duration, "mean_price")
    )


def summary_table(arm: str, effects: List[Dict], metrics: Dict, rows: List[Dict]) -> List[Dict]:
    """One row per battery config: the headline numbers for traders and policy makers."""
    table = []
    for (power, duration), m in metrics.items():
        price = _price_effect(effects, arm, power, duration)
        cycles = [
            r["bess_equivalent_cycles_mean"]
            for r in rows
            if (r["arm"], r["power_mw"], r["duration_h"]) == (arm, power, duration)
        ]
        profit = float(m["battery_profit"].mean())
        at_baseline = float(m["battery_profit_at_baseline_prices"].mean())
        consumer = float(m["delta_consumer_cost"].mean())
        table.append(
            {
                "power_mw": power,
                "duration_h": duration,
                "mean_price_delta": price["mean_delta"],
                "mean_price_ci": (price["ci_low"], price["ci_high"]),
                "consumer_cost_delta": consumer,
                "consumer_cost_pct": 100.0 * consumer / float(m["baseline_consumer_cost"].mean()),
                "seeds_lower_cost": f"{int((m['delta_consumer_cost'] < 0).sum())}/{len(m['delta_consumer_cost'])}",
                "generator_profit_delta": float(m["delta_generator_profit"].mean()),
                "battery_profit": profit,
                "profit_per_mw": profit / power,
                "profit_per_mwh": profit / (power * duration),
                "cycles": float(np.mean(cycles)) if cycles else float("nan"),
                "utilisation_pct": 100.0 * float(m["utilisation"].mean()),
                "cannibalisation_pct": 100.0 * float(m["cannibalisation"].mean()) / at_baseline if at_baseline else float("nan"),
                "loss_day_pct": 100.0 * float(m["loss_day_share"].mean()),
                "peak_price_delta": float(m["delta_peak_price"].mean()),
                "offpeak_price_delta": float(m["delta_offpeak_price"].mean()),
                "scarcity_pp": float(m["delta_scarcity_pp"].mean()),
                "peak_shaving_mw": float(m["peak_shaving_mw"].mean()),
            }
        )
    return table


def key_findings(arm: str, effects: List[Dict], metrics: Dict) -> List[str]:
    """Plain-language headlines, every number taken from the table the dashboard also shows."""
    table = summary_table(arm, effects, metrics, rows=[])  # cycles unused here
    label = lambda r: f"{r['power_mw']:g} MW / {r['duration_h']:g} h"  # noqa: E731
    largest = max(table, key=lambda r: (r["power_mw"], r["duration_h"]))
    low, high = largest["mean_price_ci"]
    findings = [
        f"Largest battery ({label(largest)}): mean price {largest['mean_price_delta']:+.2f} "
        f"(95% CI {low:+.2f} to {high:+.2f}), consumer cost {largest['consumer_cost_delta']:+,.0f}/day "
        f"({largest['consumer_cost_pct']:+.1f}%), generator profit {largest['generator_profit_delta']:+,.0f}/day, "
        f"battery profit {largest['battery_profit']:+,.0f}/day."
    ]
    best = max(table, key=lambda r: r["profit_per_mw"])
    worst = min(table, key=lambda r: r["profit_per_mw"])
    findings.append(
        f"Revenue per MW is highest for {label(best)} ({best['profit_per_mw']:,.1f}/MW/day) "
        f"and lowest for {label(worst)} ({worst['profit_per_mw']:,.1f}/MW/day)."
    )
    cannibal = max(table, key=lambda r: r["cannibalisation_pct"])
    findings.append(
        f"Price impact eats into revenue: {label(cannibal)} keeps {100 - cannibal['cannibalisation_pct']:.0f}% "
        f"of what it would earn as a price taker ({cannibal['cannibalisation_pct']:.0f}% cannibalised)."
    )
    lowers_peak = sum(r["peak_price_delta"] < 0 for r in table)
    raises_residual_peak = sum(r["peak_shaving_mw"] < 0 for r in table)
    findings.append(
        f"Peak-hour price falls in {lowers_peak} of {len(table)} configs; the residual-load peak rises in "
        f"{raises_residual_peak} (the battery charges into the demand peak)."
    )
    always = sum(r["seeds_lower_cost"].split("/")[0] == r["seeds_lower_cost"].split("/")[1] for r in table)
    findings.append(f"Consumer cost is lower in every seed for {always} of {len(table)} configs.")
    riskiest = max(table, key=lambda r: r["loss_day_pct"])
    findings.append(
        f"Battery loses money on up to {riskiest['loss_day_pct']:.0f}% of days ({label(riskiest)})."
    )
    return findings
