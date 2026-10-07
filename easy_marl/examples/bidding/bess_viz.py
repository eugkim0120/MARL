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

from easy_marl.examples.bidding.bess_experiment import (
    DELTA_METRICS,
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


def _pairs(rows: List[Dict], arm: str) -> List[Tuple[Dict, Dict]]:
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
    for r, base in _pairs(rows, arm):
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
        for r, base in _pairs(rows, arm)
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


def generator_output_change(out_dir, arm: str) -> Dict[Tuple[float, float], List[float]]:
    """Per (power, duration): mean over seeds of each generator's daily MWh minus its same-seed baseline."""
    rows = load_results(out_dir)
    daily: Dict[str, np.ndarray] = {}
    per_config: Dict[Tuple[float, float], List[np.ndarray]] = {}
    for r, base in _pairs(rows, arm):
        for config_id in (r["config_id"], base["config_id"]):
            if config_id not in daily:
                generators = _hourly_dispatch(load_dispatch(out_dir, config_id))["generators"]
                daily[config_id] = generators.sum(axis=1)
        delta = daily[r["config_id"]] - daily[base["config_id"]]
        per_config.setdefault((r["power_mw"], r["duration_h"]), []).append(delta)
    return {key: np.mean(deltas, axis=0).tolist() for key, deltas in sorted(per_config.items())}


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


def fig_heatmaps(effects: List[Dict], arm: str):
    plt = _pyplot()
    fig, axes = plt.subplots(1, len(HEATMAP_METRICS), figsize=(3.3 * len(HEATMAP_METRICS), 3.4))
    for ax, (metric, label) in zip(axes, HEATMAP_METRICS):
        powers, durations, means, significant = effect_grid(effects, arm, metric)
        limit = np.nanmax(np.abs(means)) or 1.0
        ax.imshow(means, cmap="RdBu_r", vmin=-limit, vmax=limit, origin="lower", aspect="auto")
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
        ax.set_title(label, fontsize=10)
    axes[0].set_ylabel("Power (MW)")
    fig.suptitle(f"Paired change vs no battery ({arm}); * = 95% CI excludes 0", fontsize=11)
    fig.tight_layout()
    return fig


def fig_per_seed(rows: List[Dict], effects: List[Dict], arm: str):
    plt = _pyplot()
    pairs = _pairs(rows, arm)
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


def fig_generator_change(out_dir, arm: str):
    plt = _pyplot()
    change = generator_output_change(out_dir, arm)
    configs = sorted(change, key=lambda key: (key[1], key[0]))
    n_generators = len(next(iter(change.values())))
    width = 0.8 / n_generators
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(configs) + 2), 3.8))
    ax.axhline(0.0, color="black", linewidth=0.8)
    colors = plt.cm.Greys(np.linspace(0.35, 0.8, n_generators))
    for g in range(n_generators):
        ax.bar(
            np.arange(len(configs)) + (g - (n_generators - 1) / 2) * width,
            [change[key][g] for key in configs],
            width, color=colors[g], label=f"Generator {g}",
        )
    ax.set_xticks(range(len(configs)), [_config_label(*key) for key in configs], rotation=45, ha="right")
    ax.set_ylabel("Change in daily output (MWh)")
    ax.legend(fontsize=7)
    ax.set_title(f"Generator output displaced by the battery ({arm})", fontsize=11)
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
h1 { margin-bottom: 4px; } h2 { margin-top: 40px; border-bottom: 1px solid #d0d7de; padding-bottom: 4px; }
h3 { margin-top: 28px; } p.note { color: #52606d; max-width: 80ch; margin-top: 2px; }
.cards { display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0; }
.card { border: 1px solid #d0d7de; border-radius: 8px; padding: 10px 16px; min-width: 120px; }
.card .value { font-size: 1.6rem; font-weight: 600; } .card .label { color: #52606d; font-size: 0.85rem; }
img { max-width: 100%; height: auto; display: block; margin: 8px 0; }
"""


def _largest_config(battery_rows: List[Dict], arm: str) -> Tuple[float, float]:
    """Biggest power, then longest duration, among the arm's battery runs."""
    return max(
        (r["power_mw"], r["duration_h"]) for r in battery_rows if r["arm"] == arm
    )


def _section(title: str, note: str, figure_html: str) -> str:
    return f"<h3>{html.escape(title)}</h3><p class=\"note\">{html.escape(note)}</p>{figure_html}"


def build_dashboard(out_dir, name: str = "dashboard.html") -> Path:
    out_dir = Path(out_dir)
    rows = load_results(out_dir)
    convergence = load_convergence(out_dir)
    effects = paired_effects(rows)
    arms = sorted({r["arm"] for r in rows})
    battery_rows = [r for r in rows if r["power_mw"] is not None]
    flagged = [r for r in battery_rows if r["bess_equivalent_cycles_mean"] < MIN_USEFUL_CYCLES]

    parts = [
        f"<h1>BESS sweep: {html.escape(out_dir.name)}</h1>",
        '<p class="note">Battery added next to pretrained generators; every effect is a '
        "per-seed difference against the same seed without a battery.</p>",
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
        parts += [
            f"<h2>Arm: {html.escape(arm)}</h2>",
            _section(
                "Effect heatmaps",
                "Mean paired difference per battery size and duration. Blue is lower than "
                "the no-battery market, red is higher.",
                _embed(fig_heatmaps(effects, arm), f"effect heatmaps {arm}"),
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
            _section(
                "Battery behaviour",
                "Whether the learned battery actually arbitrages: cycles, profit and the price it buys and sells at.",
                _embed(fig_battery(rows, arm), f"battery behaviour {arm}"),
            ),
        ]
        power, duration = _largest_config(battery_rows, arm)
        parts += [
            _section(
                "Dispatch",
                f"Who produces each hour with and without the largest battery ({_config_label(power, duration)}); "
                "battery charging is drawn below zero.",
                _embed(fig_dispatch(out_dir, arm, power, duration), f"hourly dispatch {arm}"),
            ),
            _section(
                "Generator output displaced",
                "Change in each generator's daily energy sold, battery minus same-seed baseline.",
                _embed(fig_generator_change(out_dir, arm), f"generator output change {arm}"),
            ),
        ]
    if convergence:
        parts.append("<h2>Pretraining</h2>")
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
        + "</body></html>"
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
