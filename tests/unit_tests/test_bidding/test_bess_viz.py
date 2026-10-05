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
from easy_marl.examples.bidding.bess_viz import (
    build_dashboard,
    effect_grid,
    hourly_deltas,
)

HOURS = 4
SEEDS = (1, 2, 3)
POWERS = (10, 25)
DURATIONS = (1, 4)


def write_config(out_dir, config, price_shift, hourly_shift):
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


def write_run(out_dir, arms=("frozen",), with_baseline=True):
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
                write_config(out_dir, SweepConfig(None, None, seed, arm), 0.0, 0.0)
            for power in POWERS:
                for duration in DURATIONS:
                    # Larger batteries lower the price more; seeds differ a little.
                    shift = -(power / 10.0) - 0.1 * seed
                    write_config(
                        out_dir,
                        SweepConfig(power, duration, seed, arm),
                        shift,
                        shift,
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


class TestDashboard:
    def test_that_dashboard_has_every_section_and_figure(self, tmp_path):
        write_run(tmp_path)

        path = build_dashboard(tmp_path)

        html = path.read_text()
        assert path.name == "dashboard.html"
        for section in (
            "Effect heatmaps",
            "Per-seed effects",
            "Hourly price profile",
            "Battery behaviour",
            "Pretraining convergence",
        ):
            assert section in html
        assert html.count("<img") == 5

    def test_that_every_arm_gets_its_own_section(self, tmp_path):
        write_run(tmp_path, arms=("frozen", "adaptive"))

        html = build_dashboard(tmp_path).read_text()

        assert "Arm: frozen" in html
        assert "Arm: adaptive" in html

    def test_that_an_empty_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_dashboard(tmp_path)
