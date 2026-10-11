# BESS follow-up: handoff notes

Snapshot of work done outside the package so a fresh (cloud) session can continue. Nothing here is imported by `easy_marl`.

## State of the experiment

- Design: 3 pretrained, frozen generators plus 1 PPO battery, day-ahead 24 h bidding, paired battery-minus-same-seed-baseline analysis with bootstrap 95% CIs, all outputs in relative units.
- Sweep: 20 battery configs (5/10/25/50/75 MW x 1/2/4/8 h) x 10 seeds (42-51), frozen arm. Results are in `results/bess_full/` (report.md, results.csv, paired_effects.csv, dashboard.html). Per-run dispatch is under `outputs/` locally and is not committed.
- Seed 49 did not converge in generator pretraining.

## Two research questions (confirmed by the user)

- A. Can RL agents learn the right trading behaviour?
- B. What do batteries do to the market/economy?
- REMIT/compliance is out of scope.

## Open decisions (need the user)

1. Run a CMA-ES open-loop benchmark on all 200 batteries (about 25 min on 8 cores). Today only 4 cases are benchmarked (74/84/89/97% of the best schedule).
2. Whether to compute a Nash benchmark for the generators. Today only the at-cost price is used as a reference; generators learned prices of about 2.1x that.
3. SoC carry-over redesign: the battery resets to 50% SoC every day and pays a terminal penalty (80/MWh below 50%). The user questioned this. Proposal: multi-day episodes (for example 7 days), SoC in the observation, penalty only at episode end. No code written.
4. Scenario realism ladder S0-S4 (see `pages/bess_sim_map.html`): S1 SoC carry-over plus wear cost, S2 weekly load shape, S3 solar with weather draw, S4 adaptive generators. The user has not picked a direction.
5. Keep the best round's policy rather than the last: all battery sizes score lower in round 5 than in round 4.

## Metric cleanup still pending

The `MIN_USEFUL_CYCLES = 0.6` flag in `bess_experiment.py` is a misleading diagnostic. Replace it with full-power hours (discharge MWh / MW; measured range 0.9-5.1 h/day) in `results/bess_full/report.md`, the dashboard, and the drafts. The drafts here have not been updated for this.

## Layout

- `drafts/`: paper draft, related work, LinkedIn post.
- `study_page/`: the two-question page (`study.html` with `@@RUNS@@` replaced by `data.json` and `@@AGG@@` replaced by `agg.json`). Published as the "Learning to Trade Power" artifact.
- `run_canvas/`: Figma-style run canvas (`app.html` with `@@DATA@@` = `study_page/data.json` and `@@PAGES@@` = `pages.json`).
- `pages/`: simulator map, dashboard proposal, and the Design-type dashboard canvas sources.
- `scripts/`: exploratory analysis scripts (price-maker bound, cycling probes, spread, hourly profiles). They were run from a scratch directory with hard-coded paths and may need path edits. The script that built `data.json` and `agg.json` was run inline and was not saved.

## Data schema (relative units only)

- `data.json`: `{base:{seed:{price,demand,gen[3],bids[3],conv,tol,converged}}, runs:[{id,p,h,s,price,offer,sellq,chg,dis,soc,gen,curve[5],m:{yld,fph,cyc,cap,dp,dbill,dgp,dstd,socend}}], probe:[{variant,p,h,s,fph,net_vs_control,secs}]}`
- `agg.json`: `{cfg:[{p,h,cons,gen,bat,net,dp[mean,lo,hi],dstd,dbill,buy_lt_sell,hourly[24]}], gens:[{s,learned,comp,ratio,markup[3]}], spot:[...]}`

## Environment notes

- Use `.venv/bin/python`; `python` and `python3` are not on PATH on the original host.
- Push to `origin` (eugkim0120/MARL) only, never `upstream` (wd7512/MARL).
