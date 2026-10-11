

def pretrain_dir(out_dir, seed: int) -> Path:
    return Path(out_dir) / "pretrain" / f"s{seed}"


def _generator_env(params: Dict, seed: int, index: int) -> MARLElectricityMarketEnv:
    return MARLElectricityMarketEnv(
        agents=[], params=params, seed=seed, agent_index=index, observer_name=DEFAULT_OBS
    )


def _load_generators(
    directory: Path, params: Dict, seed: int, round_number: Optional[int] = None
) -> List[PPOAgent]:
    """Load the final pretrained generators, or their checkpoint after a 1-based round."""
    agents = []
    for i in range(N_GENERATORS):
        if round_number is None:
            path = directory / f"agent_{i}.zip"
        else:
            path = directory / "training" / f"round_{round_number}" / f"agent_{i}.zip"
        agents.append(PPOAgent.from_bytes(path.read_bytes(), _generator_env(params, seed, i)))
    return agents


def _bid_schedules(agents: List[PPOAgent], params: Dict) -> np.ndarray:
    """Deterministic, clipped action of every agent on the evaluation demand profile."""
    env = MARLElectricityMarketEnv(
        agents=agents, params=params, seed=EVAL_SEED, observer_name=DEFAULT_OBS
    )
    env.reset(seed=EVAL_SEED)
    return np.array(
        [
            np.clip(agent.act(env._get_obs(i).copy()), -1.0, 1.0)
            for i, agent in enumerate(agents)
        ]
    )


def get_pretrained(seed: int, preset: Dict, out_dir) -> List[PPOAgent]:
    """Train the generators alone once per seed, cache them, and record how settled they are.

    Convergence is the mean absolute change of the generators' deterministic bid
    schedules between consecutive rounds.
    """
    rounds = preset["pretrain_rounds"]
    if rounds < 2:
        raise ValueError(
            f"pretrain_rounds must be at least 2 to measure convergence, got {rounds}."
        )
    directory = pretrain_dir(out_dir, seed)
    params = make_default_params(N=N_GENERATORS, T=HOURS, day_ahead=True)

    if not (directory / "convergence.json").exists():
        agents, _ = parallel_train(
            N=N_GENERATORS,
            num_rounds=rounds,
            timesteps_per_agent=preset["pretrain_timesteps_per_agent"],
            seed=seed,
            save_dir=str(directory / "training"),
            verbose=False,
            param_func=partial(make_default_params, day_ahead=True),
            update_probability=preset["update_probability"],
        )
        for i, agent in enumerate(agents):
            (directory / f"agent_{i}.zip").write_bytes(agent.save_to_bytes())

        schedules = [
            _bid_schedules(_load_generators(directory, params, seed, r), params)
            for r in range(1, rounds + 1)
        ]
        changes = [float(np.abs(b - a).mean()) for a, b in zip(schedules, schedules[1:])]
        info = {
            "policy_change_per_round": changes,
            "tolerance": preset["pretrain_change_tol"],
            "converged": bool(changes[-1] <= preset["pretrain_change_tol"]),
        }
        # Written last: its presence marks the cache as complete.
        (directory / "convergence.json").write_text(json.dumps(info, indent=2))

    return _load_generators(directory, params, seed)


def run_config(config: SweepConfig, preset: Dict, out_dir: Path) -> Dict:
    config_dir = out_dir / config.config_id
    config_dir.mkdir(parents=True, exist_ok=True)
    generators = get_pretrained(config.seed, preset, out_dir)
    n_agents = config.n_agents
    params = config.param_func()(N=n_agents, T=HOURS)

    if config.is_baseline and config.arm == "frozen":
        agents = generators
    else:
        if config.is_baseline:
            initial_agents = generators
        else:
            battery = init_agents(n_agents, params, config.seed)[-1]
            initial_agents = generators + [battery]
        frozen_agents = range(N_GENERATORS) if config.arm == "frozen" else ()
        agents, _ = parallel_train(
            N=n_agents,
            num_rounds=preset["num_rounds"],
            timesteps_per_agent=preset["timesteps_per_agent"],
            seed=config.seed,
            save_dir=str(config_dir / "training"),
            verbose=False,
            param_func=config.param_func(),
            update_probability=preset["update_probability"],
            initial_agents=initial_agents,
            frozen_agents=frozen_agents,
        )

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
# Market-wide metrics that exist with and without a battery, so a paired difference is defined.
DELTA_METRICS = [
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
]
PLOT_METRICS = [
    ("mean_price", "Mean price"),
    ("price_std", "Price std (all hours)"),
    ("daily_spread_mean", "Daily max-min price spread"),
    ("consumer_cost_mean", "Consumer cost per day"),
    ("generator_profit_total_mean", "Total generator profit per day"),
    ("bess_profit_mean", "Battery profit per day"),
]
EFFECT_COLUMNS = [
    ("mean_price", "Mean price"),
    ("price_std", "Price std"),
    ("daily_spread_mean", "Daily spread"),
    ("consumer_cost_mean", "Consumer cost"),
    ("generator_profit_total_mean", "Gen profit"),
    ("loss_of_load_mwh_mean", "Unserved MWh"),
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


def load_convergence(out_dir) -> Dict[int, Dict]:
    info = {}
    for path in sorted(Path(out_dir).glob("pretrain/s*/convergence.json")):
        info[int(path.parent.name[1:])] = json.loads(path.read_text())
    return info


def bootstrap_ci(values, n_boot: int = 10_000, level: float = 0.95, seed: int = 0):
    """Percentile bootstrap interval for the mean."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    tail = (1.0 - level) / 2.0 * 100.0
    low, high = np.percentile(means, [tail, 100.0 - tail])
    return float(low), float(high)


def paired_effects(rows: List[Dict], exclude_flagged: bool = False) -> List[Dict]:
    """Battery minus same-seed, same-arm baseline for every market metric.

    With exclude_flagged, batteries that cycled less than MIN_USEFUL_CYCLES are dropped.
    """
    baselines = {(r["arm"], r["seed"]): r for r in rows if r["power_mw"] is None}
    groups: Dict = {}
    for r in rows:
        if r["power_mw"] is None:
            continue
        if exclude_flagged and r["bess_equivalent_cycles_mean"] < MIN_USEFUL_CYCLES:
            continue
        baseline = baselines.get((r["arm"], r["seed"]))
        if baseline is None:
            raise ValueError(
                f"No baseline for {r['config_id']} (arm {r['arm']!r}, seed {r['seed']})."
            )
        groups.setdefault((r["arm"], r["power_mw"], r["duration_h"]), []).append(
            (r, baseline)
        )

    effects = []
    for (arm, power, duration), pairs in sorted(groups.items()):
        for metric in DELTA_METRICS:
            deltas = [r[metric] - base[metric] for r, base in pairs]
            if len(deltas) > 1:
                low, high = bootstrap_ci(deltas)
            else:
                low = high = float("nan")
            effects.append(
                {
                    "arm": arm,
                    "power_mw": power,
                    "duration_h": duration,
                    "metric": metric,
                    "n": len(deltas),
                    "mean_delta": float(np.mean(deltas)),
                    "ci_low": low,
                    "ci_high": high,
                    "significant": bool(low > 0 or high < 0),
                    "excluding_flagged": exclude_flagged,
                }
            )
    return effects


def _summarise(values):
    values = np.array([v for v in values if v is not None], dtype=np.float64)
    if len(values) == 0:
        return None, None
    return float(values.mean()), float(values.std())


def _suffix(arm: str) -> str:
    return "" if arm == "frozen" else f"_{arm}"


def plot_results(rows: List[Dict], out_dir: Path, arm: str) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r["arm"] == arm]
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
        ax.set_title(f"{label} ({arm})")
        ax.legend(title="Duration")
        fig.tight_layout()
        path = out_dir / f"{key}{_suffix(arm)}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_effects(effects: List[Dict], out_dir: Path, arm: str) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    effects = [e for e in effects if e["arm"] == arm]
    powers = sorted({e["power_mw"] for e in effects})
    durations = sorted({e["duration_h"] for e in effects})
    paths = []

    for key, label in PLOT_METRICS:
        if key not in DELTA_METRICS:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axhline(0.0, color="black", linestyle="--")
        for duration in durations:
            points = {
                e["power_mw"]: e
                for e in effects
                if e["metric"] == key and e["duration_h"] == duration
            }
            xs = [p for p in powers if p in points]
            means = np.array([points[p]["mean_delta"] for p in xs])
            lows = np.nan_to_num(means - np.array([points[p]["ci_low"] for p in xs]))
            highs = np.nan_to_num(np.array([points[p]["ci_high"] for p in xs]) - means)
            ax.errorbar(
                xs, means, yerr=[lows, highs], marker="o", capsize=3, label=f"{duration:g} h"
            )
        ax.set_xlabel("Battery power (MW)")
        ax.set_ylabel(f"Change in {label.lower()}")
        ax.set_title(f"Paired change vs no battery ({arm}), 95% CI")
        ax.legend(title="Duration")
        fig.tight_layout()
        path = out_dir / f"delta_{key}{_suffix(arm)}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths


def _absolute_table(rows: List[Dict]) -> List[str]:
    groups = {}
    for r in rows:
        groups.setdefault((r["power_mw"], r["duration_h"]), []).append(r)

    def sort_key(item):
        (power, duration), _ = item
        return (power is not None, power or 0, duration or 0)

    columns = EFFECT_COLUMNS[:5] + [
        ("bess_profit_mean", "BESS profit"),
        ("bess_equivalent_cycles_mean", "BESS cycles"),
        ("loss_of_load_mwh_mean", "Unserved MWh"),
    ]
    lines = [
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
    return lines


def _effects_table(effects: List[Dict]) -> List[str]:
    cells_by_config: Dict = {}
    for e in effects:
        cells_by_config.setdefault((e["power_mw"], e["duration_h"]), {})[e["metric"]] = e
    lines = [
        "| Power (MW) | Duration (h) | Seeds | " + " | ".join(c[1] for c in EFFECT_COLUMNS) + " |",
        "|" + "---|" * (3 + len(EFFECT_COLUMNS)),
    ]
    for (power, duration), by_metric in sorted(cells_by_config.items()):
        cells = []
        for key, _ in EFFECT_COLUMNS:
            e = by_metric[key]
            cell = f"{e['mean_delta']:+.2f}"
            if not np.isnan(e["ci_low"]):
                cell += f" [{e['ci_low']:+.2f}, {e['ci_high']:+.2f}]"
            cells.append(cell + (" *" if e["significant"] else ""))
        n = next(iter(by_metric.values()))["n"]
        lines.append(f"| {power:g} | {duration:g} | {n} | " + " | ".join(cells) + " |")
    return lines


def write_report(
    rows: List[Dict],
    out_dir: Path,
    plot_paths: Dict[str, List[Path]],
    effects: List[Dict],
    effects_clean: List[Dict],
    convergence: Dict[int, Dict],
) -> Path:
    lines = ["# BESS sweep results", ""]

    if convergence:
        lines += [
            "## Pretraining",
            "",
            "Generators are trained alone first. Policy change is the mean absolute change of "
            "their deterministic bid schedule (actions in [-1, 1]) between the last two rounds.",
            "",
            "| Seed | Policy change per round | Converged |",
            "|---|---|---|",
        ]
        for seed, info in sorted(convergence.items()):
            series = ", ".join(f"{c:.3f}" for c in info["policy_change_per_round"])
            lines.append(f"| {seed} | {series} | {info['converged']} |")
        lines.append("")

    for arm in sorted({r["arm"] for r in rows}):
        arm_rows = [r for r in rows if r["arm"] == arm]
        flagged = sorted(
            r["config_id"]
            for r in arm_rows
            if r["power_mw"] is not None
            and r["bess_equivalent_cycles_mean"] < MIN_USEFUL_CYCLES
        )
        lines += [
            f"## Arm: {arm}",
            "",
            "Mean over seeds (± std across seeds). Per-config values are averages over the "
            "evaluation demand episodes.",
            "",
            *_absolute_table(arm_rows),
            "",
            "### Paired effects (battery minus same-seed baseline)",
            "",
            "Mean difference with a 95% bootstrap interval across seeds; * marks an interval "
            "that excludes zero.",
            "",
            *_effects_table([e for e in effects if e["arm"] == arm]),
            "",
            f"### Batteries below {MIN_USEFUL_CYCLES:g} equivalent cycles "
            f"({len(flagged)} of {sum(r['power_mw'] is not None for r in arm_rows)})",
            "",
            ", ".join(flagged) if flagged else "none",
            "",
            "### Paired effects excluding those batteries",
            "",
            *_effects_table([e for e in effects_clean if e["arm"] == arm]),
            "",
            *[f"![{p.stem}]({p.name})" for p in plot_paths[arm]],
            "",
        ]
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

    effects = paired_effects(rows)
    effects_clean = paired_effects(rows, exclude_flagged=True)
    effect_rows = effects + effects_clean
    if effect_rows:
        with open(out_dir / "paired_effects.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(effect_rows[0].keys()))
            writer.writeheader()
            writer.writerows(effect_rows)

    plot_paths = {
        arm: plot_results(rows, out_dir, arm) + plot_effects(effects, out_dir, arm)
        for arm in sorted({r["arm"] for r in rows})
    }
    return write_report(
        rows, out_dir, plot_paths, effects, effects_clean, load_convergence(out_dir)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="train and evaluate every config (resumable)")
    run.add_argument("--preset", choices=sorted(PRESETS), required=True)
    run.add_argument("--out", help="output directory (default outputs/bess_sweep/<preset>)")
    run.add_argument("--powers", type=float, nargs="+")
    run.add_argument("--durations", type=float, nargs="+")
    run.add_argument("--seeds", type=int, nargs="+")
    run.add_argument("--arms", choices=ARMS, nargs="+")

    agg = sub.add_parser("aggregate", help="write results.csv, plots and report.md")
    agg.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.command == "aggregate":
        print(f"Report: {aggregate(args.out)}")
        return

    preset = dict(PRESETS[args.preset])
    for key in ("powers", "durations", "seeds", "arms"):
        if getattr(args, key):
            preset[key] = getattr(args, key)
    out_dir = Path(args.out or os.path.join("outputs", "bess_sweep", args.preset))
    configs = build_configs(
        preset["powers"], preset["durations"], preset["seeds"], preset["arms"]
    )
    run_sweep(configs, preset, out_dir)
    print(f"Report: {aggregate(out_dir)}")


if __name__ == "__main__":
    main()
