# What does a learning battery do to a small electricity market? A paired multi-agent reinforcement learning study

Draft for the github.io page. Full frozen-arm sweep: 5 battery sizes by 4 durations, 10 seeds. The adaptive arm (generators learning alongside the battery) has not been run. All results are relative (percentages, indices and per-MW yields) because the simulator's prices and money are made-up units with no real-world scale. All numbers come from `outputs/bess_sweep/full` (copied to `results/bess_full/` in the fork `eugkim0120/MARL`) and were produced by the code on its `main` branch.

## Summary

I add a battery to a small simulated day-ahead electricity market and measure how it changes prices, consumer cost and generator profit, comparing each run with the same market, demand and random seed without the battery. The generators learn to bid by reinforcement learning and are trained before the battery arrives, so the comparison isolates what the battery does. Across 20 battery configurations (5 to 75 MW, 1 to 8 hours) and 10 seeds, the battery lowers the mean price by at most 1.7 percent, lowers consumer cost by at most 2.8 percent, and lowers generator profit in every configuration, by 1.3 to 8.4 percent. The saving is mostly a transfer from generators: they lose 1.0 to 2.7 times what the battery earns, and consumers save more than the battery earns in only two of 20 configurations (25 MW with 4 and 8 hours). The battery's revenue per MW falls as it grows, from a yield of about 9 to 12 percent at 5 MW to about 2 to 4 percent at 50 and 75 MW. A large share of the large and long batteries barely trade (73 of 200 trained batteries cycle fewer than 0.6 times a day), so the trend across size and duration is partly a statement about what this learner finds, not only about the economics. The limits are listed in section 4.

## 1. Question

Grid batteries earn money from the price spread between hours. If they are big enough, their own trading narrows that spread. Three groups care about that for different reasons: the battery owner (does revenue scale with size and duration?), consumers (does the bill fall?) and generators (who loses?). Theory says that when generators behave strategically, storage can move welfare around in ways that do not align the incentives of these groups (Sioshansi, 2014). I want a simulation where each group's outcome can be read directly, with the effect of the battery measured against a clean baseline.

## 2. Setup

Market. Three generators with equal capacity and ascending constant marginal costs (cheap, mid, dear), each 1/3 of total capacity. Hourly demand follows a daily shape; on the evaluation days peak demand reaches about 90 percent of total generator capacity, so the system is tight at the peak. Prices are set by merit-order uniform-price clearing under a price cap. A day is one decision: each agent commits to bids for all 24 hours at once (day-ahead).

Agents. Every agent is a PPO policy (Stable-Baselines3). Generators choose a bid for each hour. The battery submits, for each hour, an offer to sell and a price-sensitive bid to buy. The battery starts each day at 50 percent state of charge and is charged a fixed penalty price for ending below that.

Training. For each seed the three generators are first trained alone for 20 rounds until their bid schedules stop changing materially (mean absolute change per round at most 0.15 in the last round; 9 of 10 seeds met this, seed 49 did not and is kept in the results). The no-battery market is those generators evaluated alone. For each battery size and duration, a new battery agent is added next to the pretrained generators and trained for 5 rounds of 10,000 days, with the generators frozen (the primary "frozen" arm). Evaluation replays 20 seeded demand days, identical across all runs, with deterministic policies.

Design. For each seed, the effect of a battery is the difference between the battery run and the no-battery run of the same seed. I report the mean over ten seeds with a bootstrap 95 percent confidence interval. Because the generators are the same policies in both runs, every difference comes from the battery's dispatch. A battery that cycles fewer than 0.6 times a day is flagged as lightly used; I show results with and without those runs.

Grid. Power 5, 10, 25, 50, 75 MW (about 3 to 50 percent of total generator capacity); duration 1, 2, 4, 8 hours; 10 seeds. That is 200 trained batteries and 10 baselines.

## 3. Results (frozen arm, 10 seeds)

Units. Every figure is relative to the same arm's no-battery market. Price change, consumer cost change and generator profit change are percent of the no-battery value. Battery profit is a yield per MW: the percent of what 1 MW sold flat all day would earn at the no-battery mean price. Across seeds the no-battery mean price varies by about 4.4 percent (SD).

Paired effects, mean difference as percent of the no-battery mean (95 percent bootstrap interval across seeds; * = interval excludes zero):

| Battery | Mean price | Consumer cost | Generator profit | Price std | Battery yield per MW (%) | Cycles/day |
|---|---|---|---|---|---|---|
| 5 MW / 1 h | -0.08 | -0.14 | -1.31 * | -0.83 | 9.2 | 2.86 |
| 5 MW / 2 h | -0.14 | -0.24 | -1.58 * | -0.82 | 9.3 | 1.73 |
| 5 MW / 4 h | -0.22 | -0.37 * | -2.03 * | -0.86 | 11.3 | 0.95 |
| 5 MW / 8 h | -0.26 | -0.46 * | -2.39 * | -1.19 | 11.6 | 0.63 |
| 10 MW / 1 h | -0.26 | -0.31 | -2.49 * | -1.02 | 8.5 | 3.09 |
| 10 MW / 2 h | +0.09 | +0.02 | -1.98 * | -1.65 * | 8.3 | 1.67 |
| 10 MW / 4 h | -0.12 | -0.22 | -2.81 * | -1.23 | 10.3 | 0.94 |
| 10 MW / 8 h | -0.15 | -0.41 * | -3.48 * | -1.09 | 10.8 | 0.61 |
| 25 MW / 1 h | -0.30 | -0.23 | -3.77 * | -1.81 * | 5.9 | 2.62 |
| 25 MW / 2 h | -0.76 * | -0.92 * | -5.13 * | -2.45 * | 5.5 | 1.59 |
| 25 MW / 4 h | -1.66 * | -2.41 * | -7.62 * | -0.61 | 5.3 | 0.81 |
| 25 MW / 8 h | -1.72 * | -2.80 * | -8.43 * | -1.40 | 5.2 | 0.49 |
| 50 MW / 1 h | -1.43 * | -1.72 * | -7.63 * | -0.06 | 3.8 | 1.69 |
| 50 MW / 2 h | -1.46 * | -1.75 * | -7.02 * | -0.82 | 3.1 | 0.91 |
| 50 MW / 4 h | -0.80 * | -1.30 * | -6.99 * | -3.48 * | 4.1 | 0.48 |
| 50 MW / 8 h | -0.75 * | -1.25 * | -6.54 * | -3.82 * | 3.4 | 0.21 |
| 75 MW / 1 h | -1.10 * | -1.41 * | -6.88 * | -1.52 | 2.2 | 0.90 |
| 75 MW / 2 h | -1.30 * | -1.81 * | -7.53 * | -1.70 | 2.6 | 0.62 |
| 75 MW / 4 h | -0.42 * | -0.97 * | -6.22 * | -4.80 * | 2.5 | 0.32 |
| 75 MW / 8 h | -0.19 | -0.60 * | -5.28 * | -4.13 * | 2.4 | 0.11 |

Findings.

1. Generators pay, consumers gain a little, the battery earns least of the three. Generator profit falls significantly in all 20 configurations (1.3 to 8.4 percent). The mean price falls significantly in 10 of 20, consumer cost in 14 of 20. Generator loss is 1.0 to 2.7 times the battery's profit (summed across seeds). Consumers save more than the battery earns only for 25 MW / 4 h and 25 MW / 8 h (1.5 and 1.7 times); in the others they save 0 to 0.9 times its profit.
2. The biggest price and bill effects are at 25 MW with 4 to 8 hours (mean price -1.7 percent, consumer cost -2.4 to -2.8 percent) and at 50 MW with 1 to 2 hours (-1.4 to -1.5 percent price). Price volatility falls most for the large, long batteries (50 and 75 MW with 4 and 8 hours: -3.5 to -4.8 percent), and the daily price spread falls 6 to 7.5 percent for the same batteries.
3. Returns to scale are poor. Yield per MW is about 9 to 12 percent at 5 MW and 2 to 4 percent at 50 and 75 MW. A 50 MW / 2 h battery keeps 63 percent of the revenue it would make as a price taker and loses the rest to its own price impact; it loses money on up to 10 percent of evaluation days.
4. The battery does not reliably shave the peak. The peak of net demand (demand plus battery charge minus discharge) rises in 12 of 20 configurations because the learned battery charges into the demand peak. Peak-hour prices still fall in 12 of 20. I have not identified that mechanism.
5. Consumer cost falls in every one of the 10 seeds for only 1 of 20 configurations, so the sign of the consumer effect is seed-dependent for most of the grid.

Excluding flagged batteries. Dropping runs below 0.6 cycles per day removes all of the 25 MW / 8 h, 50 MW / 8 h and 75 MW / 4 h and 8 h data and most of 50 MW / 4 h (1 of 10 seeds remains), so the comparison there is not usable. Where enough seeds remain, the price and cost effects keep their sign and get somewhat larger (for example 50 MW / 2 h, 7 seeds: mean price -1.9 percent, generator profit -8.7 percent). The full flagged and unflagged tables are in `report.md`.

## 4. What not to read into this

- Lightly used batteries. 73 of the 200 trained batteries cycle fewer than 0.6 times a day. All ten 25 MW / 8 h batteries do, as do all ten at 50 MW / 8 h, 75 MW / 4 h and 75 MW / 8 h, and nine of ten at 50 MW / 4 h. These batteries do charge and discharge in roughly equal amounts, so they are doing arbitrage, but the volume is small: a learned battery moves about 70 to 100 MWh a day almost regardless of its size. I checked whether this is a limit of the learner or of the market on four configurations (25 MW / 8 h with seeds 43 and 47, 75 MW / 4 h and 50 MW / 2 h with seed 43). A tuned PPO variant (larger rollouts, no entropy bonus, lower initial action noise) raised daily revenue net of the terminal penalty by up to about 35 percent over the sweep setting on some of them and lowered it on seed 47, and raised cycles only to 0.5 to 1.0 per day. An open-loop CMA-ES search over the battery's 48 day-ahead parameters, cleared in the real market with price impact included, reached net revenue within roughly 10 percent below to 35 percent above the sweep-setting PPO result and cycled at a similar level (0.36 to 0.90 per day). So on these four cases the low cycling is mostly what the market pays for once the battery's own price impact and the terminal-energy penalty are counted, not a failure to learn. This is a small spot check with one search run per case, CMA-ES can stop in a poor local optimum on this rough landscape, and I have not run it across the sweep. A price-taker linear program that ignores price impact would suggest far more cycling (about 0.95 per day for 25 MW / 8 h), which is why the price-maker comparison matters.
- Frozen generators. In this arm the generators cannot respond. It estimates the short-run effect before anyone adapts. Long-run effects need the adaptive arm, which I have not run, and where generator profits may recover; I do not know by how much.
- One training run per battery, 10 seeds, 20 evaluation days. The confidence intervals cover seed variation in generators and demand, not variation in the battery's own training. With 9 of 10 seeds converged in pretraining and 5 rounds of battery training, some of the noise is training noise.
- The terminal penalty price (80 per MWh) is below the price cap (90), which slightly favours dumping stored energy at the cap. It was not binding in the probes I ran, but I did not re-check it for the large batteries.
- Relative numbers only. The simulator's prices and money have no real-world units, so absolute figures are not reported here; the raw CSVs in the repo keep them. A percent yield is not a return on capital.
- Stylised market: three generators with constant marginal cost, inelastic demand, no ramping, no ancillary services, no renewables. The results say how the mechanism behaves, not what a specific grid will do.
- Three existing unit tests in the upstream repo (`test_agents.py`) fail independently of this work.

## 5. Related work

Search-based and incomplete; see `related_work.md` for the verification status of each item.

- Sioshansi (2014), "When energy storage reduces social welfare", Energy Economics 41: with perfectly competitive generation, storage cannot reduce welfare; with strategic generators or generator-owned storage it can; large storage lowers consumer cost and generator profit. Our distributional pattern is the one this theory describes, but our generators are learned bidders, not analytic strategic firms.
- Cruise, Flatley and Zachary (2016; EJOR 2018), "Impact of storage competition on energy markets": at the Cournot-Nash equilibrium several large price-making stores erode their own profits. Our falling yield per MW is the single-learner analogue.
- RL bidding and dispatch for storage and generators is a large literature (for example a MATD3 study of storage strategies in market models, and several single-agent battery bidding studies). I have not read these in full and I do not claim novelty beyond the paired market-level measurement.
- Trade press and operator reports from CAISO and ERCOT describe falling volatility and rising mid-day prices as batteries enter. These are context for direction, not validation.

## 6. Reproducing

```
git checkout main   # eugkim0120/MARL
uv run python -m easy_marl.examples.bidding.bess_experiment run --preset full
uv run python -m easy_marl.examples.bidding.bess_viz --out outputs/bess_sweep/full
```

The run saves each configuration's agents and full hourly dispatch (`dispatch.npz`), so every figure can be rebuilt without retraining (`bess_experiment replay`). The full sweep took roughly 5 hours of wall time, run as three parallel groups of seeds. The repo keeps the report, dashboard and CSVs under `results/bess_full/`; the 280 MB of per-config agents and dispatch stay local.

## 7. Next steps

1. A non-learning benchmark battery (optimal arbitrage against the baseline prices) to separate economics from learner quality. This is the most important one given finding 1 of section 4.
2. Adaptive arm, with generators learning alongside the battery against a no-battery control trained for the same number of rounds.
3. Several training seeds per battery, so the intervals include the battery's own learning variance.
4. Turn this into a proper paper after the benchmark and the related-work reading above.
