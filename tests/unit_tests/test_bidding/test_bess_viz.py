"""
Unit tests for the battery sweep dashboard.
"""

import json
from dataclasses import asdict

import numpy as np
import pytest

from easy_marl.examples.bidding.bess_experiment import (
    CSV_METRICS,
    SweepConfig,
    load_results,
    paired_effects,
)
from easy_marl.examples.bidding.bess_analysis import (
    config_metrics,
    dispatch_profile,
    effect_grid,
    generation_by_hour,
    generation_change_by_hour,
    key_findings,
    metric_grid,
    summary_table,
    hourly_deltas,
    load_dispatch,
)
from easy_marl.examples.bidding.bess_viz import build_dashboard

HOURS = 4
EPISODES = 2
GENERATORS = 3
SEEDS = (1, 2, 3)
POWERS = (10, 25)
DURATIONS = (1, 4)


DEMAND = [40.0, 60.0, 80.0, 100.0]
GENERATOR_COST = [10.0, 20.0, 30.0]


def write_dispatch(config_dir, config):
    """Known dispatch for hand-checked numbers.

    Baseline price in hour h is 50 + h. Generator g sells 10*(g+1) MW, minus power/10 MW each
    with a battery. The battery charges power/2 MW in the first half of the day and discharges
    power/2 MW in the second half; it pushes the price by power/50 up while charging and down
    while discharging, and sits at 25% state of charge.
    """
    has_battery = config.power_mw is not None
    n_agents = GENERATORS + int(has_battery)
    shape = (EPISODES, HOURS, n_agents)
    hours = np.arange(HOURS)
    prices = 50.0 + hours
    q_cleared = np.zeros(shape)
    for g in range(GENERATORS):
        q_cleared[:, :, g] = 10.0 * (g + 1) - (config.power_mw / 10.0 if has_battery else 0.0)
    arrays = {
        "demand": np.tile(DEMAND, (EPISODES, 1)),
        "bids": np.zeros(shape),
        "q_offered": np.zeros(shape),
        "q_cleared": q_cleared,
        "rewards": np.zeros(shape),
        "generator_cost": np.array(GENERATOR_COST),
        "generator_capacity": np.array([50.0, 50.0, 50.0]),
    }
    if has_battery:
        charging = hours < HOURS // 2
        prices = prices + np.where(charging, 1.0, -1.0) * config.power_mw / 50.0
        half = config.power_mw / 2.0
        arrays["bess_charge"] = np.tile(np.where(charging, half, 0.0), (EPISODES, 1))
        arrays["bess_discharge"] = np.tile(np.where(charging, 0.0, half), (EPISODES, 1))
        arrays["soc"] = np.full((EPISODES, HOURS), 0.25 * config.power_mw * config.duration_h)
    arrays["market_prices"] = np.tile(prices, (EPISODES, 1))
    np.savez_compressed(config_dir / "dispatch.npz", **arrays)


def write_config(out_dir, config, price_shift, hourly_shift, with_dispatch=True):
    """Write a metrics.json whose values are a known shift away from a baseline of 50."""
    metrics = {k: 50.0 + price_shift for k in CSV_METRICS}
    metrics["mean_hourly_price"] = [50.0 + hourly_shift + h for h in range(HOURS)]
    metrics["bess_equivalent_cycles_mean"] = 1.0
    if config.power_mw is None:
        for k in CSV_METRICS:
            if k.startswith("bess_"):
                metrics[k] = None
    (out_dir / config.config_id).mkdir(parents=True)
    (out_dir / config.config_id / "metrics.json").write_text(
        json.dumps({"config": asdict(config), "metrics": metrics})
    )
    if with_dispatch:
        write_dispatch(out_dir / config.config_id, config)


def write_run(out_dir, arms=("frozen",), with_baseline=True, with_dispatch=True):
    for seed in SEEDS:
        pre = out_dir / "pretrain" / f"s{seed}"
        pre.mkdir(parents=True)
        (pre / "convergence.json").write_text(
            json.dumps(
                {
                    "policy_change_per_round": [0.4, 0.2, 0.1],
                    "tolerance": 0.15,
                    "converged": True,
                }
            )
        )
        for arm in arms:
            if with_baseline:
                write_config(
                    out_dir, SweepConfig(None, None, seed, arm), 0.0, 0.0, with_dispatch
                )
            for power in POWERS:
                for duration in DURATIONS:
                    # Larger batteries lower the price more; seeds differ a little.
                    shift = -(power / 10.0) - 0.1 * seed
                    write_config(
                        out_dir,
                        SweepConfig(power, duration, seed, arm),
                        shift,
                        shift,
                        with_dispatch,
                    )


class TestHourlyDeltas:
    def test_that_hourly_delta_is_battery_minus_same_seed_baseline(self, tmp_path):
        write_run(tmp_path)

        deltas = hourly_deltas(tmp_path, arm="frozen")

        # Power 10 shifts every hour by -(1 + 0.1 * seed), so the seed mean is -1.2.
        assert deltas[(10, 1)] == pytest.approx([-1.2] * HOURS)
        assert deltas[(25, 4)] == pytest.approx([-2.7] * HOURS)

    def test_that_a_missing_baseline_raises(self, tmp_path):
        write_run(tmp_path, with_baseline=False)

        with pytest.raises(ValueError, match="baseline"):
            hourly_deltas(tmp_path, arm="frozen")


class TestEffectGrid:
    def test_that_grid_holds_mean_delta_and_significance_per_cell(self, tmp_path):
        write_run(tmp_path)
        effects = paired_effects(load_results(tmp_path))

        powers, durations, means, significant = effect_grid(
            effects, arm="frozen", metric="mean_price"
        )

        assert powers == [10, 25]
        assert durations == [1, 4]
        assert means.shape == (2, 2)
        assert means[0, 0] == pytest.approx(-1.2)
        assert means[1, 1] == pytest.approx(-2.7)
        # Every seed has a negative delta, so every interval excludes zero.
        assert significant.all()

    def test_that_an_unknown_metric_raises(self, tmp_path):
        write_run(tmp_path)
        effects = paired_effects(load_results(tmp_path))

        with pytest.raises(ValueError, match="metric"):
            effect_grid(effects, arm="frozen", metric="not_a_metric")


class TestDispatch:
    def test_that_missing_dispatch_points_at_replay(self, tmp_path):
        write_run(tmp_path, with_dispatch=False)

        with pytest.raises(FileNotFoundError, match="replay"):
            load_dispatch(tmp_path, "baseline_s1")

    def test_that_profile_averages_seeds_and_episodes_for_battery_and_baseline(self, tmp_path):
        write_run(tmp_path)

        profile = dispatch_profile(tmp_path, arm="frozen", power=25, duration=4)

        assert profile["demand"] == pytest.approx(DEMAND)
        assert profile["baseline"]["generators"].shape == (GENERATORS, HOURS)
        assert profile["baseline"]["generators"][:, 0] == pytest.approx([10.0, 20.0, 30.0])
        assert profile["battery"]["generators"][:, 0] == pytest.approx([7.5, 17.5, 27.5])
        assert profile["battery"]["discharge"] == pytest.approx([0.0, 0.0, 12.5, 12.5])
        assert profile["battery"]["charge"] == pytest.approx([12.5, 12.5, 0.0, 0.0])
        assert profile["baseline"]["discharge"] is None

    def test_that_an_unknown_config_raises(self, tmp_path):
        write_run(tmp_path)

        with pytest.raises(ValueError, match="config"):
            dispatch_profile(tmp_path, arm="frozen", power=99, duration=4)


class TestGenerationChangeByHour:
    def test_that_rows_are_each_generator_then_battery_net_output(self, tmp_path):
        write_run(tmp_path)

        change = generation_change_by_hour(tmp_path, arm="frozen")

        rows = change[(25, 4)]
        assert rows.shape == (GENERATORS + 1, HOURS)
        # Each generator sells power/10 MW less every hour; the battery nets discharge minus charge.
        assert rows[:GENERATORS] == pytest.approx(np.full((GENERATORS, HOURS), -2.5))
        assert rows[GENERATORS] == pytest.approx([-12.5, -12.5, 12.5, 12.5])


class TestGenerationByHour:
    def test_that_levels_are_seed_means_per_plant_with_a_baseline_entry(self, tmp_path):
        write_run(tmp_path)

        levels = generation_by_hour(tmp_path, arm="frozen")

        assert levels["baseline"][:, 0] == pytest.approx([10.0, 20.0, 30.0, 0.0])
        assert levels[(25, 4)][:, 0] == pytest.approx([7.5, 17.5, 27.5, -12.5])


class TestConfigMetrics:
    """Hand-computed from the write_dispatch fixture for the 25 MW / 4 h battery (see its docstring)."""

    @pytest.fixture
    def metrics(self, tmp_path):
        write_run(tmp_path)
        return config_metrics(tmp_path, arm="frozen")[(25, 4)]

    def test_that_every_seed_gets_a_value(self, metrics):
        assert metrics["battery_profit"].shape == (len(SEEDS),)
        assert metrics["daily_profit"].shape == (len(SEEDS), EPISODES)

    def test_that_battery_profit_and_cannibalisation_compare_realised_with_baseline_prices(self, metrics):
        # 12.5 MW sold at 51.5 and 52.5, bought at 50.5 and 51.5 -> 25; at baseline prices 50.
        assert metrics["battery_profit"] == pytest.approx([25.0] * len(SEEDS))
        assert metrics["battery_profit_at_baseline_prices"] == pytest.approx([50.0] * len(SEEDS))
        assert metrics["cannibalisation"] == pytest.approx([25.0] * len(SEEDS))
        assert metrics["daily_profit"] == pytest.approx(25.0)
        assert metrics["loss_day_share"] == pytest.approx(0.0)

    def test_that_welfare_components_are_battery_minus_baseline_per_day(self, metrics):
        assert metrics["delta_consumer_cost"] == pytest.approx([-40.0] * len(SEEDS))
        assert metrics["delta_generator_profit"] == pytest.approx([-945.0] * len(SEEDS))
        assert metrics["delta_generation_cost"] == pytest.approx([-600.0] * len(SEEDS))
        # The fixture's supply does not equal demand, so the identity leaves this residual.
        assert metrics["welfare_residual"] == pytest.approx([-40.0 + 945.0 + 600.0 - 25.0] * len(SEEDS))

    def test_that_generator_profit_change_is_reported_per_plant(self, metrics):
        by_plant = metrics["delta_generator_profit_by_plant"]
        assert by_plant.shape == (len(SEEDS), GENERATORS)
        assert by_plant[0] == pytest.approx([1245.0 - 1660.0, 2205.0 - 2520.0, 2365.0 - 2580.0])

    def test_that_peak_and_off_peak_hours_come_from_demand_rank(self, metrics):
        # With four hours the single highest-demand hour is 3 and the lowest is hour 0.
        assert metrics["delta_peak_price"] == pytest.approx([52.5 - 53.0] * len(SEEDS))
        assert metrics["delta_offpeak_price"] == pytest.approx([50.5 - 50.0] * len(SEEDS))

    def test_that_scarcity_counts_hours_at_or_above_the_baseline_p95_in_percentage_points(self, metrics):
        # Baseline p95 is 53, reached in hour 3: 25% of hours. With the battery the max is 52.5.
        assert metrics["delta_scarcity_pp"] == pytest.approx([-25.0] * len(SEEDS))

    def test_that_peak_shaving_is_the_drop_in_peak_residual_load(self, metrics):
        # Peak demand 100 MW becomes max(52.5, 72.5, 67.5, 87.5) = 87.5 MW.
        assert metrics["peak_shaving_mw"] == pytest.approx([12.5] * len(SEEDS))

    def test_that_schedule_is_a_fraction_of_rated_power_and_energy(self, metrics):
        assert metrics["net_frac_by_hour"][0] == pytest.approx([-0.5, -0.5, 0.5, 0.5])
        assert metrics["soc_frac_by_hour"][0] == pytest.approx([0.25] * HOURS)
        assert metrics["utilisation"] == pytest.approx([1.0] * len(SEEDS))

    def test_that_price_duration_curves_span_each_markets_min_to_max(self, metrics):
        assert metrics["pdc_base"].shape == (len(SEEDS), 101)
        assert metrics["pdc_base"][0, 0] == pytest.approx(50.0)
        assert metrics["pdc_base"][0, -1] == pytest.approx(53.0)
        assert metrics["pdc_battery"][0, 0] == pytest.approx(50.5)
        assert metrics["pdc_battery"][0, -1] == pytest.approx(52.5)


class TestMetricGrid:
    def test_that_grid_holds_mean_and_significance_of_a_seed_level_metric(self, tmp_path):
        write_run(tmp_path)
        metrics = config_metrics(tmp_path, arm="frozen")

        powers, durations, means, significant = metric_grid(metrics, "peak_shaving_mw")

        assert powers == [10, 25]
        assert durations == [1, 4]
        assert means.shape == (2, 2)
        # Peak demand 100 MW drops by half the rated power: 5 MW at 10 MW, 12.5 MW at 25 MW.
        assert means[0] == pytest.approx([5.0, 5.0])
        assert means[1] == pytest.approx([12.5, 12.5])
        assert significant.all()

    def test_that_an_unknown_metric_raises(self, tmp_path):
        write_run(tmp_path)

        with pytest.raises(ValueError, match="metric"):
            metric_grid(config_metrics(tmp_path, arm="frozen"), "nope")


class TestSummaryAndFindings:
    @pytest.fixture
    def parts(self, tmp_path):
        write_run(tmp_path)
        rows = load_results(tmp_path)
        return paired_effects(rows), config_metrics(tmp_path, arm="frozen"), rows

    def test_that_the_table_has_one_row_per_config_with_per_mw_returns(self, parts):
        effects, metrics, rows = parts

        table = summary_table("frozen", effects, metrics, rows)

        assert len(table) == len(POWERS) * len(DURATIONS)
        row = next(r for r in table if r["power_mw"] == 25 and r["duration_h"] == 4)
        assert row["battery_profit"] == pytest.approx(25.0)
        assert row["profit_per_mw"] == pytest.approx(1.0)
        assert row["profit_per_mwh"] == pytest.approx(0.25)
        assert row["cannibalisation_pct"] == pytest.approx(50.0)
        assert row["consumer_cost_delta"] == pytest.approx(-40.0)
        assert row["consumer_cost_pct"] == pytest.approx(-100.0 * 40.0 / 14520.0)
        assert row["seeds_lower_cost"] == f"{len(SEEDS)}/{len(SEEDS)}"

    def test_that_findings_quote_the_largest_battery(self, parts):
        effects, metrics, rows = parts

        findings = key_findings("frozen", effects, metrics)

        assert any("25 MW / 4 h" in f and "-40" in f for f in findings)
        assert all(isinstance(f, str) for f in findings)


class TestDashboard:
    def test_that_dashboard_has_every_section_and_figure(self, tmp_path):
        write_run(tmp_path)

        path = build_dashboard(tmp_path)

        html = path.read_text()
        assert path.name == "dashboard.html"
        for section in (
            "Key findings",
            "Policy view",
            "Trader view",
            "Effect heatmaps",
            "Peak, off-peak and scarcity",
            "Who gains and who pays",
            "Value of storage to consumers",
            "Price distribution",
            "Generator profit by plant",
            "Per-seed effects",
            "Hourly price profile",
            "Returns on capacity",
            "Daily profit distribution",
            "Battery schedule",
            "Battery behaviour",
            "Dispatch",
            "Generation by plant",
            "Generation change by hour",
            "Pretraining convergence",
        ):
            assert section in html
        assert "Generator output displaced" not in html
        assert html.count("<img") == 16

    def test_that_dashboard_has_a_sortable_table_with_every_config(self, tmp_path):
        write_run(tmp_path)

        html = build_dashboard(tmp_path).read_text()

        assert html.count('<table class="data">') == 1
        assert html.count("<tr>") == 1 + len(POWERS) * len(DURATIONS)
        assert 'data-value="25.0"' in html
        assert "addEventListener('click'" in html

    def test_that_every_arm_gets_its_own_section(self, tmp_path):
        write_run(tmp_path, arms=("frozen", "adaptive"))

        html = build_dashboard(tmp_path).read_text()

        assert "Arm: frozen" in html
        assert "Arm: adaptive" in html

    def test_that_a_run_without_dispatch_files_raises_pointing_at_replay(self, tmp_path):
        write_run(tmp_path, with_dispatch=False)

        with pytest.raises(FileNotFoundError, match="replay"):
            build_dashboard(tmp_path)

    def test_that_an_empty_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_dashboard(tmp_path)
