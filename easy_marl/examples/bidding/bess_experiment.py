"""
Battery (BESS) sweep: 3 generators plus one PPO-trained battery, varying battery
power and duration, compared against the same market with no battery.

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

from easy_marl.src.environment import MARLElectricityMarketEnv
from easy_marl.examples.bidding.training import (
    DEFAULT_OBS,
    make_bess_params,
    make_default_params,
    parallel_train,
)

N_GENERATORS = 3
# Same demand draws for every config, so differences come from the battery, not noise.
EVAL_SEED = 10_000

PRESETS = {
    "smoke": {
        "powers": [10, 25, 50],
        "durations": [1, 4],
        "seeds": [42],
        "num_rounds": 2,
        "timesteps_per_agent": 1_000,
        "eval_episodes": 20,
    },
    "full": {
        "powers": [5, 10, 25, 50, 75],
        "durations": [1, 2, 4, 8],
        "seeds": [42, 43, 44],
        "num_rounds": 5,
        "timesteps_per_agent": 5_000,
        "eval_episodes": 20,
    },
}


@dataclass(frozen=True)
class SweepConfig:
    power_mw: Optional[float]
    duration_h: Optional[float]
    seed: int

    @property
    def is_baseline(self) -> bool:
        return self.power_mw is None

    @property
    def config_id(self) -> str:
        if self.is_baseline:
            return f"baseline_s{self.seed}"
        return f"p{self.power_mw:g}_d{self.duration_h:g}_s{self.seed}"

    @property
    def n_agents(self) -> int:
        return N_GENERATORS if self.is_baseline else N_GENERATORS + 1

    def param_func(self):
        if self.is_baseline:
            return make_default_params
        return partial(
            make_bess_params, power_mw=self.power_mw, duration_h=self.duration_h
        )


def build_configs(powers, durations, seeds) -> List[SweepConfig]:
    configs = []
    for seed in seeds:
        configs.append(SweepConfig(None, None, seed))
        for power in powers:
            for duration in durations:
                configs.append(SweepConfig(power, duration, seed))
    return configs


def evaluate_market_metrics(
    agents,
    params: Dict,
    num_episodes: int,
    seed: Optional[int],
    observer_name: str = DEFAULT_OBS,
) -> Dict:
    """Run frozen agents over seeded demand episodes and summarise market outcomes."""
    prices, demands, gen_profits = [], [], []
    charges, discharges = [], []
    loss_of_load = []

    for ep in range(num_episodes):
        env = MARLElectricityMarketEnv(
            agents=agents,
            params=params,
            seed=None if seed is None else seed + ep,
            observer_name=observer_name,
        )
        env.reset()
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(None, fixed_evaluation=True)
            done = terminated or truncated

        out = env.output
        price = out["market_prices"].astype(np.float64)
        prices.append(price)
        demands.append(out["demand"].astype(np.float64))
        q_gen = out["q_cleared"][:, : env.N_generators].astype(np.float64)
        gen_profits.append(((price[:, None] - env.c[None, :]) * q_gen).sum(axis=0))
        charge = out["bess_charge"] if env.has_bess else np.zeros(env.T)
        discharge = out["bess_discharge"] if env.has_bess else np.zeros(env.T)
        charges.append(np.asarray(charge, dtype=np.float64))
        discharges.append(np.asarray(discharge, dtype=np.float64))
        served = q_gen.sum(axis=1) + discharges[-1] - charges[-1]
        loss_of_load.append(np.maximum(demands[-1] - served, 0.0).sum())

    prices = np.array(prices)
    demands = np.array(demands)
    gen_profits = np.array(gen_profits)
    charges = np.array(charges)
    discharges = np.array(discharges)

    metrics = {
        "mean_price": float(prices.mean()),
        "price_std": float(prices.std()),
        "intraday_std_mean": float(prices.std(axis=1).mean()),
        "daily_spread_mean": float((prices.max(axis=1) - prices.min(axis=1)).mean()),
        "price_p5": float(np.percentile(prices, 5)),
        "price_p95": float(np.percentile(prices, 95)),
        "price_max": float(prices.max()),
        "mean_hourly_price": prices.mean(axis=0).tolist(),
        "consumer_cost_mean": float((prices * demands).sum(axis=1).mean()),
        "loss_of_load_mwh_mean": float(np.mean(loss_of_load)),
        "generator_profit_mean": gen_profits.mean(axis=0).tolist(),
        "generator_profit_total_mean": float(gen_profits.sum(axis=1).mean()),
        "bess_profit_mean": None,
        "bess_charge_mwh_mean": None,
        "bess_discharge_mwh_mean": None,
        "bess_equivalent_cycles_mean": None,
        "bess_mean_charge_price": None,
        "bess_mean_discharge_price": None,
    }

    if "bess" in params:
        energy = params["bess"]["power_mw"] * params["bess"]["duration_h"]
        charged_total = charges.sum()
        discharged_total = discharges.sum()
        metrics.update(
            {
                "bess_profit_mean": float(
                    (prices * (discharges - charges)).sum(axis=1).mean()
                ),
                "bess_charge_mwh_mean": float(charges.sum(axis=1).mean()),
                "bess_discharge_mwh_mean": float(discharges.sum(axis=1).mean()),
                "bess_equivalent_cycles_mean": float(
                    discharges.sum(axis=1).mean() / energy
                ),
                "bess_mean_charge_price": (
                    float((prices * charges).sum() / charged_total)
                    if charged_total > 0
                    else None
                ),
                "bess_mean_discharge_price": (
                    float((prices * discharges).sum() / discharged_total)
                    if discharged_total > 0
                    else None
                ),
            }
        )
    return metrics


def run_config(config: SweepConfig, preset: Dict, out_dir: Path) -> Dict:
    config_dir = out_dir / config.config_id
    agents, training_info = parallel_train(
        N=config.n_agents,
        num_rounds=preset["num_rounds"],
        timesteps_per_agent=preset["timesteps_per_agent"],
        seed=config.seed,
        save_dir=str(config_dir / "training"),
        verbose=False,
        param_func=config.param_func(),
    )
    params = config.param_func()(N=config.n_agents, T=24)
    metrics = evaluate_market_metrics(
        agents, params, num_episodes=preset["eval_episodes"], seed=EVAL_SEED
    )
    result = {"config": asdict(config), "preset": preset, "metrics": metrics}
    with open(config_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def run_sweep(configs: List[SweepConfig], preset: Dict, out_dir) -> None:
    """Train and evaluate every config; configs with an existing metrics.json are skipped."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, config in enumerate(configs, start=1):
        if (out_dir / config.config_id / "metrics.json").exists():
            print(f"[{i}/{len(configs)}] {config.config_id}: done, skipping")
            continue
        print(f"[{i}/{len(configs)}] {config.config_id}: training", flush=True)
        result = run_config(config, preset, out_dir)
        m = result["metrics"]
        print(
            f"    mean_price={m['mean_price']:.2f} price_std={m['price_std']:.2f} "
            f"spread={m['daily_spread_mean']:.2f}",
            flush=True,
        )


CSV_METRICS = [
    "mean_price",
    "price_std",
    "intraday_std_mean",
    "daily_spread_mean",
    "price_p5",
    "price_p95",
    "price_max",
    "consumer_cost_mean",
    "loss_of_load_mwh_mean",
    "generator_profit_total_mean",
    "bess_profit_mean",
    "bess_charge_mwh_mean",
    "bess_discharge_mwh_mean",
    "bess_equivalent_cycles_mean",
    "bess_mean_charge_price",
    "bess_mean_discharge_price",
]
PLOT_METRICS = [
    ("mean_price", "Mean price"),
    ("price_std", "Price std (all hours)"),
    ("daily_spread_mean", "Daily max-min price spread"),
    ("consumer_cost_mean", "Consumer cost per day"),
    ("generator_profit_total_mean", "Total generator profit per day"),
    ("bess_profit_mean", "Battery profit per day"),
]


def load_results(out_dir) -> List[Dict]:
    rows = []
    for path in sorted(Path(out_dir).glob("*/metrics.json")):
        result = json.loads(path.read_text())
        row = {"config_id": path.parent.name, **result["config"]}
        row.update({k: result["metrics"][k] for k in CSV_METRICS})
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No */metrics.json under {out_dir}")
    return rows


def _summarise(values):
    values = np.array([v for v in values if v is not None], dtype=np.float64)
    if len(values) == 0:
        return None, None
    return float(values.mean()), float(values.std())


def plot_results(rows: List[Dict], out_dir: Path) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = [r for r in rows if r["power_mw"] is None]
    battery = [r for r in rows if r["power_mw"] is not None]
    powers = sorted({r["power_mw"] for r in battery})
    durations = sorted({r["duration_h"] for r in battery})
    paths = []

    for key, label in PLOT_METRICS:
        fig, ax = plt.subplots(figsize=(6, 4))
        base_mean, base_std = _summarise(r[key] for r in baseline)
        if base_mean is not None:
            ax.axhline(base_mean, color="black", linestyle="--", label="no battery")
            if base_std:
                ax.axhspan(base_mean - base_std, base_mean + base_std, color="grey", alpha=0.2)
        for duration in durations:
            means, stds = [], []
            for power in powers:
                m, s = _summarise(
                    r[key]
                    for r in battery
                    if r["power_mw"] == power and r["duration_h"] == duration
                )
                means.append(np.nan if m is None else m)
                stds.append(0.0 if s is None else s)
            ax.errorbar(powers, means, yerr=stds, marker="o", capsize=3, label=f"{duration:g} h")
        ax.set_xlabel("Battery power (MW)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(title="Duration")
        fig.tight_layout()
        path = out_dir / f"{key}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths


def write_report(rows: List[Dict], out_dir: Path, plot_paths: List[Path]) -> Path:
    groups = {}
    for r in rows:
        groups.setdefault((r["power_mw"], r["duration_h"]), []).append(r)

    def sort_key(item):
        (power, duration), _ = item
        return (power is not None, power or 0, duration or 0)

    columns = [
        ("mean_price", "Mean price"),
        ("price_std", "Price std"),
        ("daily_spread_mean", "Daily spread"),
        ("consumer_cost_mean", "Consumer cost"),
        ("generator_profit_total_mean", "Gen profit"),
        ("bess_profit_mean", "BESS profit"),
        ("bess_equivalent_cycles_mean", "BESS cycles"),
        ("loss_of_load_mwh_mean", "Unserved MWh"),
    ]
    lines = [
        "# BESS sweep results",
        "",
        "Mean over seeds (± std across seeds when more than one). "
        "Per-config values are averages over the evaluation demand episodes.",
        "",
        "| Power (MW) | Duration (h) | Seeds | " + " | ".join(c[1] for c in columns) + " |",
        "|" + "---|" * (3 + len(columns)),
    ]
    for (power, duration), group in sorted(groups.items(), key=sort_key):
        cells = []
        for key, _ in columns:
            mean, std = _summarise(r[key] for r in group)
            if mean is None:
                cells.append("–")
            elif len(group) > 1:
                cells.append(f"{mean:.2f} ± {std:.2f}")
            else:
                cells.append(f"{mean:.2f}")
        power_cell = "none" if power is None else f"{power:g}"
        duration_cell = "–" if duration is None else f"{duration:g}"
        lines.append(
            f"| {power_cell} | {duration_cell} | {len(group)} | " + " | ".join(cells) + " |"
        )
    lines.append("")
    lines += [f"![{p.stem}]({p.name})" for p in plot_paths]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def aggregate(out_dir) -> Path:
    out_dir = Path(out_dir)
    rows = load_results(out_dir)
    with open(out_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_paths = plot_results(rows, out_dir)
    return write_report(rows, out_dir, plot_paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="train and evaluate every config (resumable)")
    run.add_argument("--preset", choices=sorted(PRESETS), required=True)
    run.add_argument("--out", help="output directory (default outputs/bess_sweep/<preset>)")
    run.add_argument("--powers", type=float, nargs="+")
    run.add_argument("--durations", type=float, nargs="+")
    run.add_argument("--seeds", type=int, nargs="+")

    agg = sub.add_parser("aggregate", help="write results.csv, plots and report.md")
    agg.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.command == "aggregate":
        print(f"Report: {aggregate(args.out)}")
        return

    preset = dict(PRESETS[args.preset])
    for key in ("powers", "durations", "seeds"):
        if getattr(args, key):
            preset[key] = getattr(args, key)
    out_dir = Path(args.out or os.path.join("outputs", "bess_sweep", args.preset))
    configs = build_configs(preset["powers"], preset["durations"], preset["seeds"])
    run_sweep(configs, preset, out_dir)
    print(f"Report: {aggregate(out_dir)}")


if __name__ == "__main__":
    main()
