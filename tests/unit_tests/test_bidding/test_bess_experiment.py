"""
Unit tests for the battery (BESS) experiment pipeline.
"""

import json

import numpy as np
import pytest

from easy_marl.src.agents import SimpleAgent
from easy_marl.examples.bidding import bess_experiment
from easy_marl.examples.bidding.bess_experiment import (
    SweepConfig,
    build_configs,
    evaluate_market_metrics,
    run_sweep,
)
from easy_marl.examples.bidding.training import (
    evaluate_agents,
    make_bess_params,
    make_default_params,
)

T = 4
POWER = 20.0
DURATION = 2.0
BESS_INDEX = 3

PARAMS = {
    "N_generators": 3,
    "T": T,
    "demand_profile": [60.0] * T,
    "capacities": [50.0, 50.0, 50.0],
    "costs": [10.0, 20.0, 30.0],
    "max_bid_delta": 10.0,
    "lambda_bid_penalty": 0.0,
    "bess": {
        "power_mw": POWER,
        "duration_h": DURATION,
        "efficiency_rt": 0.81,
        "initial_soc_frac": 0.5,
        "bid_ref": 25.0,
    },
}


class FixedAgent:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)

    def fixed_act_function(self, deterministic=True):
        return lambda obs: self.action


def fixed_agents(bess_action):
    return [SimpleAgent() for _ in range(3)] + [FixedAgent(bess_action)]


class TestMakeBessParams:
    def test_that_generators_and_demand_match_the_no_battery_baseline(self):
        baseline = make_default_params(N=3, T=24)
        params = make_bess_params(N=4, T=24, power_mw=POWER, duration_h=DURATION)

        for key in ("N_generators", "demand_profile", "capacities", "costs"):
            assert params[key] == baseline[key]
        assert params["bess"]["power_mw"] == POWER
        assert params["bess"]["duration_h"] == DURATION

    def test_that_default_battery_starts_empty_so_it_cannot_sell_free_energy(self):
        params = make_bess_params(N=4, T=24, power_mw=POWER, duration_h=DURATION)
        assert params["bess"]["initial_soc_frac"] == 0.0

        metrics = evaluate_market_metrics(
            fixed_agents([1.0, -10.0]), {**PARAMS, "bess": params["bess"]},
            num_episodes=1, seed=None,
        )
        assert metrics["bess_discharge_mwh_mean"] == 0.0
        assert metrics["bess_profit_mean"] == 0.0


class TestEvaluateWithBattery:
    def test_that_battery_profit_is_price_times_net_energy_sold(self):
        results = evaluate_agents(fixed_agents([1.0, -10.0]), PARAMS)

        # Hour 0: offer at 15 is marginal after the 10 generator, sells 10 at 15.
        # Hour 1: remaining 8 MWh deliverable sells at 20 (20 generator marginal).
        assert results["mean_quantities"][BESS_INDEX] == pytest.approx(18.0, rel=1e-5)
        assert results["mean_profits"][BESS_INDEX] == pytest.approx(
            15.0 * 10.0 + 20.0 * 8.0, rel=1e-5
        )


class TestEvaluateMarketMetrics:
    def test_that_constant_prices_give_zero_volatility_and_known_costs(self):
        metrics = evaluate_market_metrics(
            [SimpleAgent() for _ in range(3)],
            {k: v for k, v in PARAMS.items() if k != "bess"},
            num_episodes=2,
            seed=None,
        )
        # Demand 60 is met by the 10 generator (50) and 10 from the 20 generator.
        assert metrics["mean_price"] == pytest.approx(20.0)
        assert metrics["price_std"] == pytest.approx(0.0)
        assert metrics["daily_spread_mean"] == pytest.approx(0.0)
        assert metrics["consumer_cost_mean"] == pytest.approx(20.0 * 60.0 * T)
        assert metrics["loss_of_load_mwh_mean"] == pytest.approx(0.0)
        assert metrics["generator_profit_total_mean"] == pytest.approx(10.0 * 50.0 * T)
        assert metrics["bess_profit_mean"] is None

    def test_that_charging_battery_reports_cycles_and_purchase_price(self):
        metrics = evaluate_market_metrics(
            fixed_agents([-1.0, 10.0]), PARAMS, num_episodes=1, seed=None
        )
        # Charges 20 MW in hour 0 (store 20 -> 38), then 2/0.9 more in hour 1 (full).
        charged = 20.0 + (40.0 - 38.0) / 0.9
        assert metrics["bess_charge_mwh_mean"] == pytest.approx(charged, rel=1e-5)
        assert metrics["bess_discharge_mwh_mean"] == pytest.approx(0.0)
        assert metrics["bess_equivalent_cycles_mean"] == pytest.approx(0.0)
        assert metrics["bess_mean_charge_price"] == pytest.approx(20.0)
        assert metrics["bess_profit_mean"] == pytest.approx(-20.0 * charged, rel=1e-5)


class TestSweep:
    def test_that_configs_are_baseline_plus_power_by_duration_grid_per_seed(self):
        configs = build_configs(powers=[10, 25], durations=[1, 4], seeds=[42, 43])

        assert len(configs) == (1 + 2 * 2) * 2
        assert SweepConfig(None, None, 42) in configs
        assert SweepConfig(25, 4, 43) in configs
        assert len({c.config_id for c in configs}) == len(configs)

    def test_that_finished_configs_are_skipped_on_resume(self, tmp_path, monkeypatch):
        configs = build_configs(powers=[10], durations=[1], seeds=[42])
        for config in configs:
            config_dir = tmp_path / config.config_id
            config_dir.mkdir()
            (config_dir / "metrics.json").write_text(json.dumps({"done": True}))

        def fail_if_called(*args, **kwargs):
            raise AssertionError("training ran for a finished config")

        monkeypatch.setattr(bess_experiment, "parallel_train", fail_if_called)
        run_sweep(configs, bess_experiment.PRESETS["smoke"], tmp_path)

    @pytest.mark.parametrize("preset_name", sorted(bess_experiment.PRESETS))
    def test_that_every_agent_trains_every_round(self, preset_name, tmp_path, monkeypatch):
        preset = bess_experiment.PRESETS[preset_name]
        assert preset["update_probability"] == 1.0

        captured = {}

        class StopAfterCapture(Exception):
            pass

        def capture(**kwargs):
            captured.update(kwargs)
            raise StopAfterCapture

        monkeypatch.setattr(bess_experiment, "parallel_train", capture)
        with pytest.raises(StopAfterCapture):
            bess_experiment.run_config(SweepConfig(10, 1, 42), preset, tmp_path)
        assert captured["update_probability"] == 1.0
