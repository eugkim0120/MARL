"""
Dashboard for the battery (BESS) sweep: one self-contained HTML page built from the
files a sweep leaves behind (metrics.json per config and pretrain/*/convergence.json).

Usage:
    python -m easy_marl.examples.bidding.bess_viz --out outputs/bess_sweep/smoke
"""

import argparse
import base64
import html
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from easy_marl.examples.bidding.bess_analysis import (
    baseline_hourly,
    config_metrics,
    dispatch_profile,
    effect_grid,
    generation_by_hour,
    generation_change_by_hour,
    hourly_deltas,
    key_findings,
    metric_grid,
    paired_rows,
    summary_table,
)
from easy_marl.examples.bidding.bess_experiment import (
    MIN_USEFUL_CYCLES,
    bootstrap_ci,
    load_convergence,
    load_results,
    paired_effects,
)

HEATMAP_METRICS = [
    ("mean_price", "Mean price"),
    ("price_std", "Price std"),
    ("daily_spread_mean", "Daily spread"),
    ("consumer_cost_mean", "Consumer cost / day"),
    ("generator_profit_total_mean", "Generator profit / day"),
]
STRIP_METRICS = [
    ("mean_price", "Mean price"),
    ("price_std", "Price std"),
    ("daily_spread_mean", "Daily spread"),
]
IMAGE_DPI = 110


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _duration_colors(durations):
    plt = _pyplot()
    return {d: plt.cm.tab10(i % 10) for i, d in enumerate(durations)}


def _config_label(power, duration) -> str:
    return f"{power:g} MW / {duration:g} h"


def _draw_grid(ax, powers, durations, means, significant, title, cmap="RdBu_r"):
    limit = np.nanmax(np.abs(means)) or 1.0
    ax.imshow(means, cmap=cmap, vmin=-limit, vmax=limit, origin="lower", aspect="auto")
    for i in range(len(powers)):
        for j in range(len(durations)):
            value = means[i, j]
            text = (f"{value:+,.0f}" if abs(value) >= 100 else f"{value:+.2f}") + (
                "*" if significant[i, j] else ""
            )
            color = "white" if abs(value) > 0.6 * limit else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)
    ax.set_xticks(range(len(durations)), [f"{d:g} h" for d in durations])
    ax.set_yticks(range(len(powers)), [f"{p:g}" for p in powers])
    ax.set_xlabel("Duration")
    ax.set_title(title, fontsize=10)


def fig_heatmaps(effects: List[Dict], arm: str):
    plt = _pyplot()
    fig, axes = plt.subplots(1, len(HEATMAP_METRICS), figsize=(3.3 * len(HEATMAP_METRICS), 3.4))
    for ax, (metric, label) in zip(axes, HEATMAP_METRICS):
        _draw_grid(ax, *effect_grid(effects, arm, metric), label)
    axes[0].set_ylabel("Power (MW)")
    fig.suptitle(f"Paired change vs no battery ({arm}); * = 95% CI excludes 0", fontsize=11)
    fig.tight_layout()
    return fig


PRICE_SHAPE_METRICS = [
    ("delta_peak_price", "Peak-hour price"),
    ("delta_offpeak_price", "Off-peak-hour price"),
    ("delta_scarcity_pp", "Scarcity hours (pp of hours)"),
    ("peak_shaving_mw", "Peak load shaved (MW)"),
]


def fig_price_shape(metrics: Dict, arm: str):
    plt = _pyplot()
    fig, axes = plt.subplots(1, len(PRICE_SHAPE_METRICS), figsize=(3.4 * len(PRICE_SHAPE_METRICS), 3.4))
    for ax, (name, label) in zip(axes, PRICE_SHAPE_METRICS):
        _draw_grid(ax, *metric_grid(metrics, name), label)
    axes[0].set_ylabel("Power (MW)")
    fig.suptitle(
        f"Where in the day the battery acts ({arm}); peak = highest-demand hours, scarcity = hours at "
        "or above the no-battery P95 price; * = 95% CI excludes 0",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def fig_per_seed(rows: List[Dict], effects: List[Dict], arm: str):
    plt = _pyplot()
    pairs = paired_rows(rows, arm)
    configs = sorted({(r["duration_h"], r["power_mw"]) for r, _ in pairs})
    fig, axes = plt.subplots(
        len(STRIP_METRICS), 1, figsize=(max(6, 0.9 * len(configs) + 2), 3.0 * len(STRIP_METRICS)),
        sharex=True,
    )
    colors = _duration_colors(sorted({d for d, _ in configs}))
    rng = np.random.default_rng(0)
    for ax, (metric, label) in zip(axes, STRIP_METRICS):
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        for x, (duration, power) in enumerate(configs):
            deltas = [
                r[metric] - base[metric]
                for r, base in pairs
                if r["power_mw"] == power and r["duration_h"] == duration
            ]
            ax.scatter(
                x + rng.uniform(-0.15, 0.15, len(deltas)), deltas, s=14, alpha=0.6,
                color=colors[duration],
            )
            low, high = bootstrap_ci(deltas) if len(deltas) > 1 else (np.nan, np.nan)
            ax.errorbar(
                x, np.mean(deltas), yerr=[[np.mean(deltas) - low], [high - np.mean(deltas)]],
                color="black", marker="D", markersize=4, capsize=4, linewidth=1.2,
            )
        ax.set_ylabel(f"Change in {label.lower()}")
    axes[-1].set_xticks(range(len(configs)), [_config_label(p, d) for d, p in configs], rotation=45, ha="right")
    axes[0].set_title(f"Per-seed paired differences ({arm}); diamond = mean with 95% CI", fontsize=11)
    fig.tight_layout()
    return fig


def fig_hourly(out_dir, arm: str):
    plt = _pyplot()
    deltas = hourly_deltas(out_dir, arm)
    durations = sorted({d for _, d in deltas})
    powers = sorted({p for p, _ in deltas})
    base = baseline_hourly(out_dir, arm)
    hours = np.arange(len(base))
    fig, axes = plt.subplots(1, 1 + len(durations), figsize=(3.6 * (1 + len(durations)), 3.4), sharex=True)
    axes[0].plot(hours, base, color="black", marker="o", markersize=3)
    axes[0].set_title("Baseline mean price by hour", fontsize=10)
    axes[0].set_ylabel("Price")
    shades = plt.cm.viridis(np.linspace(0.1, 0.9, len(powers)))
    for ax, duration in zip(axes[1:], durations):
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        for color, power in zip(shades, powers):
            ax.plot(hours, deltas[(power, duration)], color=color, label=f"{power:g} MW")
        ax.set_title(f"{duration:g} h battery: change vs baseline", fontsize=10)
        ax.legend(fontsize=7)
    for ax in axes:
        ax.set_xlabel("Hour of day")
    fig.suptitle(f"Hourly price profile ({arm})", fontsize=11)
    fig.tight_layout()
    return fig


def _by_config(rows: List[Dict], arm: str, key: str):
    groups: Dict[Tuple[float, float], List[float]] = {}
    for r in rows:
        if r["arm"] == arm and r["power_mw"] is not None and r[key] is not None:
            groups.setdefault((r["duration_h"], r["power_mw"]), []).append(r[key])
    return groups


def _stack(ax, hours, profile, title):
    plt = _pyplot()
    generators = profile["generators"]
    colors = plt.cm.Greys(np.linspace(0.35, 0.8, len(generators)))
    layers = list(generators)
    labels = [f"Generator {g}" for g in range(len(generators))]
    if profile["discharge"] is not None:
        layers.append(profile["discharge"])
        labels.append("Battery discharge")
        colors = list(colors) + ["tab:green"]
    ax.stackplot(hours, layers, labels=labels, colors=colors)
    if profile["charge"] is not None:
        ax.fill_between(hours, 0, -profile["charge"], color="tab:red", alpha=0.7, label="Battery charge")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("MW")


def fig_dispatch(out_dir, arm: str, power: float, duration: float):
    plt = _pyplot()
    profile = dispatch_profile(out_dir, arm, power, duration)
    hours = np.arange(len(profile["demand"]))
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharex=True)
    _stack(axes[0], hours, profile["baseline"], "No battery")
    _stack(axes[1], hours, profile["battery"], f"With {_config_label(power, duration)} battery")
    for ax in axes[:2]:
        ax.plot(hours, profile["demand"], color="tab:blue", linestyle="--", label="Demand")
    axes[1].legend(fontsize=7, loc="upper left", ncol=2)
    axes[2].plot(hours, profile["baseline"]["price"], color="black", label="No battery")
    axes[2].plot(hours, profile["battery"]["price"], color="tab:green", label="With battery")
    axes[2].set_title("Market price", fontsize=10)
    axes[2].set_xlabel("Hour of day")
    axes[2].set_ylabel("Price")
    axes[2].legend(fontsize=7)
    fig.suptitle(f"Hourly dispatch, {_config_label(power, duration)} ({arm}), mean over seeds and episodes", fontsize=11)
    fig.tight_layout()
    return fig


def _plant_names(n_units: int) -> List[str]:
    return [f"Generator {g}" for g in range(n_units - 1)] + ["Battery (net output)"]


def fig_generation_levels(out_dir, arm: str):
    plt = _pyplot()
    levels = generation_by_hour(out_dir, arm)
    baseline = levels.pop("baseline")
    durations = sorted({d for _, d in levels})
    powers = sorted({p for p, _ in levels})
    n_units = baseline.shape[0]
    names = _plant_names(n_units)
    fig, axes = plt.subplots(
        n_units, len(durations), figsize=(3.6 * len(durations), 2.3 * n_units),
        sharex=True, sharey="row", squeeze=False,
    )
    shades = plt.cm.viridis(np.linspace(0.1, 0.9, len(powers)))
    for u in range(n_units):
        for k, duration in enumerate(durations):
            ax = axes[u, k]
            ax.plot(baseline[u], color="black", linestyle="--", linewidth=1.4, label="No battery")
            for color, power in zip(shades, powers):
                ax.plot(levels[(power, duration)][u], color=color, label=f"{power:g} MW")
            if u == 0:
                ax.set_title(f"{duration:g} h battery", fontsize=10)
            if k == 0:
                ax.set_ylabel(f"{names[u]}\n(MW)", fontsize=8)
            if u == n_units - 1:
                ax.set_xlabel("Hour of day")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Hourly generation of each plant ({arm}), mean over seeds", fontsize=11)
    fig.tight_layout()
    return fig


def fig_generation_change(out_dir, arm: str):
    plt = _pyplot()
    change = generation_change_by_hour(out_dir, arm)
    durations = sorted({d for _, d in change})
    powers = sorted({p for p, _ in change})
    n_units = next(iter(change.values())).shape[0]
    names = _plant_names(n_units)
    fig, axes = plt.subplots(
        n_units, len(durations), figsize=(3.6 * len(durations), 2.3 * n_units),
        sharex=True, sharey="row", squeeze=False,
    )
    shades = plt.cm.viridis(np.linspace(0.1, 0.9, len(powers)))
    for u in range(n_units):
        for k, duration in enumerate(durations):
            ax = axes[u, k]
            ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
            for color, power in zip(shades, powers):
                ax.plot(change[(power, duration)][u], color=color, label=f"{power:g} MW")
            if u == 0:
                ax.set_title(f"{duration:g} h battery", fontsize=10)
            if k == 0:
                ax.set_ylabel(f"{names[u]}\nchange (MW)", fontsize=8)
            if u == n_units - 1:
                ax.set_xlabel("Hour of day")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Change in hourly generation vs no battery ({arm})", fontsize=11)
    fig.tight_layout()
    return fig


def _config_order(metrics: Dict) -> List[Tuple[float, float]]:
    return sorted(metrics, key=lambda key: (key[1], key[0]))


def _seed_mean(metrics: Dict, key, name: str) -> float:
    return float(np.mean(metrics[key][name]))


def fig_welfare(metrics: Dict, arm: str):
    """Decompose the change in consumer cost into who gains and who pays, per day."""
    plt = _pyplot()
    configs = _config_order(metrics)
    components = [
        ("delta_generator_profit", "Generator profit", "tab:gray"),
        ("delta_generation_cost", "Generation cost", "tab:orange"),
        ("battery_profit", "Battery profit", "tab:green"),
        ("welfare_residual", "Unserved / other", "lightgray"),
    ]
    fig, ax = plt.subplots(figsize=(max(7, 1.0 * len(configs) + 2), 4.2))
    x = np.arange(len(configs))
    positive = np.zeros(len(configs))
    negative = np.zeros(len(configs))
    for name, label, color in components:
        values = np.array([_seed_mean(metrics, key, name) for key in configs])
        bottom = np.where(values >= 0, positive, negative)
        ax.bar(x, values, bottom=bottom, color=color, label=label)
        positive += np.where(values >= 0, values, 0.0)
        negative += np.where(values < 0, values, 0.0)
    totals = np.array([_seed_mean(metrics, key, "delta_consumer_cost") for key in configs])
    ax.scatter(x, totals, color="black", marker="D", zorder=5, label="Change in consumer cost")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, [_config_label(*key) for key in configs], rotation=45, ha="right")
    ax.set_ylabel("Change per day vs no battery")
    ax.set_title(
        f"Who gains and who pays ({arm})\nconsumer cost change = generator profit + generation cost "
        "+ battery profit + other", fontsize=9,
    )
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def fig_consumer_value(metrics: Dict, arm: str):
    plt = _pyplot()
    durations = sorted({d for _, d in metrics})
    colors = _duration_colors(durations)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for duration in durations:
        powers = sorted(p for p, d in metrics if d == duration)
        keys = [(p, duration) for p in powers]
        saving_pct = [
            -100.0 * _seed_mean(metrics, k, "delta_consumer_cost") / _seed_mean(metrics, k, "baseline_consumer_cost")
            for k in keys
        ]
        saving_per_mw = [-_seed_mean(metrics, k, "delta_consumer_cost") / k[0] for k in keys]
        axes[0].plot(powers, saving_pct, marker="o", color=colors[duration], label=f"{duration:g} h")
        axes[1].plot(powers, saving_per_mw, marker="o", color=colors[duration], label=f"{duration:g} h")
    axes[0].set_ylabel("Saved (% of no-battery bill)")
    axes[1].set_ylabel("Saved per MW per day")
    for ax in axes:
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Battery power (MW)")
        ax.legend(title="Duration", fontsize=7)
    fig.suptitle(f"Value of storage to consumers ({arm}); a falling right panel means diminishing returns", fontsize=10)
    fig.tight_layout()
    return fig


def fig_price_duration(metrics: Dict, arm: str):
    plt = _pyplot()
    durations = sorted({d for _, d in metrics})
    powers = sorted({p for p, _ in metrics})
    shades = plt.cm.viridis(np.linspace(0.1, 0.9, len(powers)))
    fig, axes = plt.subplots(2, len(durations), figsize=(3.6 * len(durations), 6.2), sharex=True, squeeze=False)
    percentiles = np.linspace(0, 100, next(iter(metrics.values()))["pdc_base"].shape[1])
    base = next(iter(metrics.values()))["pdc_base"].mean(axis=0)
    for k, duration in enumerate(durations):
        axes[0, k].plot(percentiles, base, color="black", linestyle="--", label="No battery")
        axes[1, k].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        for color, power in zip(shades, powers):
            curve = metrics[(power, duration)]["pdc_battery"].mean(axis=0)
            axes[0, k].plot(percentiles, curve, color=color, label=f"{power:g} MW")
            axes[1, k].plot(percentiles, curve - metrics[(power, duration)]["pdc_base"].mean(axis=0), color=color)
        axes[0, k].set_title(f"{duration:g} h battery", fontsize=10)
        axes[1, k].set_xlabel("Price percentile (hours sorted cheap to dear)")
    axes[0, 0].set_ylabel("Price")
    axes[1, 0].set_ylabel("Change in price vs no battery")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Price distribution ({arm}): does the battery cut the expensive tail or fill the cheap one?", fontsize=10)
    fig.tight_layout()
    return fig


def fig_profit_by_plant(metrics: Dict, arm: str):
    plt = _pyplot()
    configs = _config_order(metrics)
    n_gen = metrics[configs[0]]["delta_generator_profit_by_plant"].shape[1]
    width = 0.8 / n_gen
    colors = plt.cm.Greys(np.linspace(0.35, 0.8, n_gen))
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(configs) + 2), 3.8))
    ax.axhline(0.0, color="black", linewidth=0.8)
    for g in range(n_gen):
        ax.bar(
            np.arange(len(configs)) + (g - (n_gen - 1) / 2) * width,
            [metrics[key]["delta_generator_profit_by_plant"][:, g].mean() for key in configs],
            width, color=colors[g], label=f"Generator {g}",
        )
    ax.set_xticks(range(len(configs)), [_config_label(*key) for key in configs], rotation=45, ha="right")
    ax.set_ylabel("Change in profit per day")
    ax.legend(fontsize=7)
    ax.set_title(f"Which generators lose margin ({arm})", fontsize=11)
    fig.tight_layout()
    return fig


def fig_trader_returns(metrics: Dict, arm: str):
    plt = _pyplot()
    durations = sorted({d for _, d in metrics})
    colors = _duration_colors(durations)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for duration in durations:
        powers = sorted(p for p, d in metrics if d == duration)
        keys = [(p, duration) for p in powers]
        series = [
            ([_seed_mean(metrics, k, "battery_profit") / k[0] for k in keys],
             [np.std(metrics[k]["battery_profit"]) / k[0] for k in keys]),
            ([_seed_mean(metrics, k, "battery_profit") / (k[0] * k[1]) for k in keys],
             [np.std(metrics[k]["battery_profit"]) / (k[0] * k[1]) for k in keys]),
            ([100 * _seed_mean(metrics, k, "cannibalisation") / _seed_mean(metrics, k, "battery_profit_at_baseline_prices")
              for k in keys], None),
        ]
        for ax, (values, spread) in zip(axes, series):
            ax.errorbar(powers, values, yerr=spread, marker="o", capsize=3, color=colors[duration], label=f"{duration:g} h")
    for ax, label in zip(axes, ("Profit per MW per day", "Profit per MWh of capacity per day", "Revenue lost to own price impact (%)")):
        ax.set_xlabel("Battery power (MW)")
        ax.set_ylabel(label)
        ax.legend(title="Duration", fontsize=7)
    axes[2].set_ylim(bottom=0)
    fig.suptitle(f"Returns on capacity ({arm}); error bars = std across seeds", fontsize=10)
    fig.tight_layout()
    return fig


def fig_schedule(out_dir, metrics: Dict, arm: str):
    plt = _pyplot()
    configs = _config_order(metrics)
    hours = np.arange(metrics[configs[0]]["net_frac_by_hour"].shape[1])
    net = np.array([metrics[k]["net_frac_by_hour"].mean(axis=0) for k in configs])
    soc = np.array([metrics[k]["soc_frac_by_hour"].mean(axis=0) for k in configs])
    labels = [_config_label(*k) for k in configs]
    fig, axes = plt.subplots(
        3, 1, figsize=(10, 2.4 + 0.32 * len(configs) * 2), sharex=True,
        gridspec_kw={"height_ratios": [2.2, len(configs), len(configs)]},
    )
    axes[0].plot(hours, baseline_hourly(out_dir, arm), color="black", marker="o", markersize=3)
    axes[0].set_ylabel("No-battery\nprice")
    extent = (-0.5, len(hours) - 0.5, -0.5, len(configs) - 0.5)
    image = axes[1].imshow(net, cmap="RdBu", vmin=-1, vmax=1, aspect="auto", origin="lower", extent=extent)
    axes[1].set_title("Net output as a fraction of rated power (red = discharging, blue = charging)", fontsize=9)
    fig.colorbar(image, ax=axes[1], pad=0.01, fraction=0.03)
    image = axes[2].imshow(soc, cmap="viridis", vmin=0, vmax=1, aspect="auto", origin="lower", extent=extent)
    axes[2].set_title("State of charge as a fraction of energy capacity", fontsize=9)
    fig.colorbar(image, ax=axes[2], pad=0.01, fraction=0.03)
    for ax in axes[1:]:
        ax.set_yticks(range(len(configs)), labels, fontsize=7)
    axes[2].set_xlabel("Hour of day")
    fig.suptitle(f"Battery schedule by hour ({arm}), mean over seeds and days", fontsize=11)
    fig.tight_layout()
    heat, line = axes[1].get_position(), axes[0].get_position()
    axes[0].set_position([heat.x0, line.y0, heat.width, line.height])
    return fig


def fig_profit_distribution(metrics: Dict, arm: str):
    plt = _pyplot()
    configs = _config_order(metrics)
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(configs) + 2), 4.0))
    ax.boxplot(
        [metrics[k]["daily_profit"].ravel() for k in configs], positions=range(len(configs)),
        showfliers=False, whis=(5, 95),
    )
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.08 * (high - low))
    for x, key in enumerate(configs):
        share = 100 * metrics[key]["loss_day_share"].mean()
        ax.text(x, ax.get_ylim()[1], f"{share:.0f}%", ha="center", va="top", fontsize=8, color="tab:red")
    ax.set_xticks(range(len(configs)), [_config_label(*k) for k in configs], rotation=45, ha="right")
    ax.set_ylabel("Battery profit per day")
    ax.set_title(f"Daily profit spread ({arm}); box = quartiles, whiskers = P5-P95, red = share of loss days", fontsize=9)
    fig.tight_layout()
    return fig


def fig_battery(rows: List[Dict], arm: str):
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    durations = sorted({d for d, _ in _by_config(rows, arm, "bess_profit_mean")})
    colors = _duration_colors(durations)

    for ax, key, label in (
        (axes[0], "bess_equivalent_cycles_mean", "Equivalent cycles / day"),
        (axes[1], "bess_profit_mean", "Battery profit / day"),
    ):
        groups = _by_config(rows, arm, key)
        for duration in durations:
            powers = sorted(p for d, p in groups if d == duration)
            means = [np.mean(groups[(duration, p)]) for p in powers]
            stds = [np.std(groups[(duration, p)]) for p in powers]
            ax.errorbar(powers, means, yerr=stds, marker="o", capsize=3,
                        color=colors[duration], label=f"{duration:g} h")
        ax.set_xlabel("Battery power (MW)")
        ax.set_ylabel(label)
        ax.legend(title="Duration", fontsize=7)
    axes[0].axhline(MIN_USEFUL_CYCLES, color="grey", linestyle=":")
    axes[0].text(axes[0].get_xlim()[0], MIN_USEFUL_CYCLES, " flag threshold", va="bottom", fontsize=7, color="grey")

    charge = _by_config(rows, arm, "bess_mean_charge_price")
    discharge = _by_config(rows, arm, "bess_mean_discharge_price")
    ax = axes[2]
    for duration, power in sorted(set(charge) & set(discharge)):
        ax.scatter(np.mean(charge[(duration, power)]), np.mean(discharge[(duration, power)]),
                   color=colors[duration], s=30 + 3 * power)
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, color="grey", linestyle=":")
    ax.set_xlabel("Mean charge price")
    ax.set_ylabel("Mean discharge price")
    ax.set_title("Above the line = buys low, sells high (size = power)", fontsize=9)
    fig.suptitle(f"Battery behaviour ({arm})", fontsize=11)
    fig.tight_layout()
    return fig


def fig_convergence(convergence: Dict[int, Dict]):
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for seed, info in sorted(convergence.items()):
        changes = info["policy_change_per_round"]
        ax.plot(range(2, len(changes) + 2), changes, marker="o", markersize=3,
                alpha=0.8, label=f"seed {seed}", linestyle="-" if info["converged"] else "--")
    ax.axhline(next(iter(convergence.values()))["tolerance"], color="black", linestyle=":", label="tolerance")
    ax.set_xlabel("Pretraining round")
    ax.set_ylabel("Mean change of bid schedule")
    ax.set_title("Pretraining convergence (dashed = not converged)", fontsize=11)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    return fig


def _embed(fig, alt: str) -> str:
    plt = _pyplot()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=IMAGE_DPI)
    plt.close(fig)
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{data}">'


def _card(label: str, value: str) -> str:
    return f'<div class="card"><div class="value">{html.escape(value)}</div><div class="label">{html.escape(label)}</div></div>'


CSS = """
body { font-family: system-ui, sans-serif; margin: 0 auto; max-width: 1280px; padding: 24px 16px; color: #1f2933; background: #fff; }
h1 { margin-bottom: 4px; } h2 { margin-top: 48px; border-bottom: 2px solid #1f2933; padding-bottom: 4px; }
h3 { margin-top: 32px; margin-bottom: 4px; color: #243b53; } h4 { margin: 24px 0 2px; color: #52606d; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 0.05em; }
p.note { color: #52606d; max-width: 80ch; margin-top: 2px; }
nav { position: sticky; top: 0; background: #fffe; padding: 8px 0; border-bottom: 1px solid #d0d7de; z-index: 2; font-size: 0.9rem; }
nav a { margin-right: 14px; color: #0b69a3; text-decoration: none; }
.cards { display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0; }
.card { border: 1px solid #d0d7de; border-radius: 8px; padding: 10px 16px; min-width: 120px; }
.card .value { font-size: 1.6rem; font-weight: 600; } .card .label { color: #52606d; font-size: 0.85rem; }
.findings { background: #f0f7ff; border-left: 4px solid #0b69a3; padding: 8px 20px; margin: 16px 0; max-width: 100ch; }
.findings li { margin: 6px 0; }
table.data { border-collapse: collapse; font-size: 0.85rem; margin: 8px 0; display: block; overflow-x: auto; }
table.data th, table.data td { border: 1px solid #d0d7de; padding: 4px 8px; text-align: right; white-space: nowrap; }
table.data th { background: #f5f7fa; cursor: pointer; position: sticky; top: 0; }
table.data tr:nth-child(even) td { background: #fafbfc; }
img { max-width: 100%; height: auto; display: block; margin: 8px 0; }
"""

SORT_SCRIPT = """
document.querySelectorAll('table.data').forEach(function (table) {
  table.querySelectorAll('th').forEach(function (th, column) {
    th.addEventListener('click', function () {
      var body = table.tBodies[0];
      var direction = th.dataset.direction === 'asc' ? 'desc' : 'asc';
      th.dataset.direction = direction;
      var rows = Array.from(body.rows);
      rows.sort(function (a, b) {
        var x = parseFloat(a.cells[column].dataset.value), y = parseFloat(b.cells[column].dataset.value);
        return direction === 'asc' ? x - y : y - x;
      });
      rows.forEach(function (row) { body.appendChild(row); });
    });
  });
});
"""

TABLE_COLUMNS = [
    ("power_mw", "Power (MW)", "{:g}"),
    ("duration_h", "Duration (h)", "{:g}"),
    ("mean_price_delta", "Mean price change", "{:+.2f}"),
    ("consumer_cost_delta", "Consumer cost change / day", "{:+,.0f}"),
    ("consumer_cost_pct", "Consumer cost change (%)", "{:+.2f}"),
    ("generator_profit_delta", "Generator profit change / day", "{:+,.0f}"),
    ("battery_profit", "Battery profit / day", "{:+,.0f}"),
    ("profit_per_mw", "Profit per MW", "{:+,.1f}"),
    ("profit_per_mwh", "Profit per MWh", "{:+,.1f}"),
    ("cycles", "Cycles / day", "{:.2f}"),
    ("utilisation_pct", "Hours active (%)", "{:.0f}"),
    ("cannibalisation_pct", "Revenue lost to own impact (%)", "{:.0f}"),
    ("loss_day_pct", "Loss days (%)", "{:.1f}"),
    ("peak_price_delta", "Peak-hour price change", "{:+.2f}"),
    ("offpeak_price_delta", "Off-peak price change", "{:+.2f}"),
    ("scarcity_pp", "Scarcity hours (pp)", "{:+.1f}"),
    ("peak_shaving_mw", "Peak load shaved (MW)", "{:+.1f}"),
]


def _summary_table_html(table_rows: List[Dict]) -> str:
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label, _ in TABLE_COLUMNS) + "<th>Seeds with lower cost</th>"
    body = []
    for row in table_rows:
        cells = []
        for key, _, fmt in TABLE_COLUMNS:
            value = row[key]
            cells.append(f'<td data-value="{value}">{html.escape(fmt.format(value))}</td>')
        wins, total = (int(part) for part in row["seeds_lower_cost"].split("/"))
        cells.append(f'<td data-value="{wins / total}">{html.escape(row["seeds_lower_cost"])}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="data"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def _largest_config(battery_rows: List[Dict], arm: str) -> Tuple[float, float]:
    """Biggest power, then longest duration, among the arm's battery runs."""
    return max(
        (r["power_mw"], r["duration_h"]) for r in battery_rows if r["arm"] == arm
    )


def _section(title: str, note: str, figure_html: str) -> str:
    return f"<h3>{html.escape(title)}</h3><p class=\"note\">{html.escape(note)}</p>{figure_html}"


def _arm_sections(out_dir: Path, arm: str, rows, effects, metrics, battery_rows) -> List[str]:
    anchor = html.escape(arm)
    power, duration = _largest_config(battery_rows, arm)
    findings = "".join(f"<li>{html.escape(text)}</li>" for text in key_findings(arm, effects, metrics))
    return [
        f'<h2 id="{anchor}">Arm: {anchor}</h2>',
        f'<div class="findings"><strong>Key findings</strong><ul>{findings}</ul></div>',
        f'<h2 id="{anchor}-policy">Policy view: prices and who pays ({anchor})</h2>',
        _section(
            "Effect heatmaps",
            "Mean paired difference per battery size and duration. Blue is lower than the no-battery market, red is higher.",
            _embed(fig_heatmaps(effects, arm), f"effect heatmaps {arm}"),
        ),
        _section(
            "Peak, off-peak and scarcity",
            "Whether the battery helps when the system is tight. Peak and off-peak are the highest- and "
            "lowest-demand hours of each day; scarcity hours are those at or above the no-battery P95 price. "
            "Peak load shaved is the drop in demand net of battery output, and can be negative when the battery "
            "charges into the demand peak.",
            _embed(fig_price_shape(metrics, arm), f"price shape {arm}"),
        ),
        _section(
            "Who gains and who pays",
            "Change in consumer cost split into generator profit, generation cost, battery profit and the remainder "
            "(unserved demand and other). Consumers gain mostly at generators' expense, not from the battery's own margin.",
            _embed(fig_welfare(metrics, arm), f"welfare decomposition {arm}"),
        ),
        _section(
            "Value of storage to consumers",
            "Savings as a share of the no-battery bill and per MW installed. Flattening curves mean diminishing returns.",
            _embed(fig_consumer_value(metrics, arm), f"consumer value {arm}"),
        ),
        _section(
            "Price distribution",
            "Price by percentile of hours, and its change against no battery, for each duration.",
            _embed(fig_price_duration(metrics, arm), f"price duration {arm}"),
        ),
        _section(
            "Generator profit by plant",
            "Which generators give up margin when the battery arrives.",
            _embed(fig_profit_by_plant(metrics, arm), f"profit by plant {arm}"),
        ),
        _section(
            "Per-seed effects",
            "Each dot is one seed's difference; the diamond is the mean with its bootstrap interval.",
            _embed(fig_per_seed(rows, effects, arm), f"per-seed effects {arm}"),
        ),
        _section(
            "Hourly price profile",
            "When in the day the battery moves the price, averaged over seeds.",
            _embed(fig_hourly(out_dir, arm), f"hourly price profile {arm}"),
        ),
        f'<h2 id="{anchor}-trader">Trader view: returns and risk ({anchor})</h2>',
        _section(
            "Returns on capacity",
            "Profit per MW and per MWh of capacity per day, and the share of price-taker revenue lost to the battery's own price impact.",
            _embed(fig_trader_returns(metrics, arm), f"trader returns {arm}"),
        ),
        _section(
            "Daily profit distribution",
            "Spread of daily profit across seeds and evaluation days, with the share of days the battery lost money.",
            _embed(fig_profit_distribution(metrics, arm), f"daily profit distribution {arm}"),
        ),
        _section(
            "Battery schedule",
            "When each battery charges and discharges, and how full it is, against the no-battery price shape.",
            _embed(fig_schedule(out_dir, metrics, arm), f"battery schedule {arm}"),
        ),
        _section(
            "Battery behaviour",
            "Whether the learned battery actually arbitrages: cycles, profit and the price it buys and sells at.",
            _embed(fig_battery(rows, arm), f"battery behaviour {arm}"),
        ),
        f'<h2 id="{anchor}-generation">Generation ({anchor})</h2>',
        _section(
            "Dispatch",
            f"Who produces each hour with and without the largest battery ({_config_label(power, duration)}); "
            "battery charging is drawn below zero.",
            _embed(fig_dispatch(out_dir, arm, power, duration), f"hourly dispatch {arm}"),
        ),
        _section(
            "Generation by plant",
            "Hourly output of each generator and the battery (discharge minus charge, so negative "
            "while charging) for every size and duration; dashed black is the market without a battery.",
            _embed(fig_generation_levels(out_dir, arm), f"generation by plant {arm}"),
        ),
        _section(
            "Generation change by hour",
            "Hour by hour, how much each generator and the battery (discharge minus charge) "
            "produce relative to the same seed without a battery, for every size and duration.",
            _embed(fig_generation_change(out_dir, arm), f"generation change by hour {arm}"),
        ),
        f'<h2 id="{anchor}-data">Data ({anchor})</h2>',
        '<p class="note">Click a column header to sort. Money is in the simulator\'s price units per day.</p>',
        _summary_table_html(summary_table(arm, effects, metrics, rows)),
    ]


def build_dashboard(out_dir, name: str = "dashboard.html") -> Path:
    out_dir = Path(out_dir)
    rows = load_results(out_dir)
    convergence = load_convergence(out_dir)
    effects = paired_effects(rows)
    arms = sorted({r["arm"] for r in rows})
    battery_rows = [r for r in rows if r["power_mw"] is not None]
    flagged = [r for r in battery_rows if r["bess_equivalent_cycles_mean"] < MIN_USEFUL_CYCLES]

    nav = "".join(
        f'<a href="#{html.escape(arm)}-{section}">{html.escape(arm)} {section}</a>'
        for arm in arms
        for section in ("policy", "trader", "generation", "data")
    )
    parts = [
        f"<h1>BESS sweep: {html.escape(out_dir.name)}</h1>",
        '<p class="note">Battery added next to pretrained generators; every effect is a '
        "per-seed difference against the same seed without a battery.</p>",
        f"<nav>{nav}{'<a href=\"#pretraining\">pretraining</a>' if convergence else ''}</nav>",
        '<div class="cards">',
        _card("seeds", str(len({r["seed"] for r in rows}))),
        _card("battery configs", str(len({(r["power_mw"], r["duration_h"]) for r in battery_rows}))),
        _card("runs", str(len(rows))),
        _card(f"batteries < {MIN_USEFUL_CYCLES:g} cycles", f"{len(flagged)} of {len(battery_rows)}"),
    ]
    if convergence:
        converged = sum(info["converged"] for info in convergence.values())
        parts.append(_card("seeds converged", f"{converged} of {len(convergence)}"))
    parts.append("</div>")

    for arm in arms:
        metrics = config_metrics(out_dir, arm)
        parts += _arm_sections(out_dir, arm, rows, effects, metrics, battery_rows)
    if convergence:
        parts.append('<h2 id="pretraining">Pretraining</h2>')
        parts.append(
            _section(
                "Pretraining convergence",
                "Mean absolute change of the generators' deterministic bid schedule between rounds.",
                _embed(fig_convergence(convergence), "pretraining convergence"),
            )
        )

    page = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>BESS sweep dashboard</title><style>{CSS}</style></head><body>"
        + "".join(parts)
        + f"<script>{SORT_SCRIPT}</script></body></html>"
    )
    path = out_dir / name
    path.write_text(page)
    return path


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True, help="sweep output directory")
    args = parser.parse_args(argv)
    print(f"Dashboard: {build_dashboard(args.out)}")


if __name__ == "__main__":
    main()
