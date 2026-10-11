# Related work: first pass

Status: search-based, not a full literature review. Searched on 2026-10-07 with WebSearch.
"Fetched" means I opened the page and read the abstract. "Snippet" means I only saw the search-result summary and have not read the paper. Nothing below has been checked against the full text, and the comparisons to our pilot are my reading, to be confirmed before anything is cited in a paper. Items marked "not verified" come from my own memory and were not found in this search.

## Closest to our question (storage changes prices, and who wins or loses)

1. Sioshansi, R. "When energy storage reduces social welfare." Energy Economics 41 (2014) 106-116.
   https://www.sciencedirect.com/science/article/abs/pii/S0140988313002223 (snippet; publisher page returned 403 to fetch)
   Snippet says: with perfectly competitive generation that does not own storage, storage cannot reduce welfare; generator-owned storage, or standalone storage facing strategic generators, can. Also says large storage smooths prices, lowering consumer cost and generator profit, so the incentives of merchant storage, consumers and generators are not aligned.
   Relevance: our pilot sees exactly the distributional pattern (consumers gain, generators lose far more than the battery earns). Our generators are learned bidders rather than analytically strategic firms, so this is the theory we should test against, not just cite.

2. "The effects of bulk electricity storage on the PJM market." Energy Systems (Springer), 2014. https://link.springer.com/article/10.1007/s12667-014-0123-7 (snippet only; fetch hit a login redirect)
   Relevance: an empirical-model study of storage effects on a real market. Authors and findings not verified; read before citing.

3. Cruise, J., Flatley, L., Zachary, S. "Impact of storage competition on energy markets." arXiv:1606.05361 (2016); European Journal of Operational Research 269(3), 2018, 998-1012. https://arxiv.org/abs/1606.05361 (fetched abstract)
   Abstract says price-making storage operators can optimise profit, generation cost or welfare, and that at the Cournot-Nash equilibrium multiple large stores collectively erode their own ability to make profits. Illustrated on Great Britain spot prices.
   Relevance: matches our cannibalisation result (revenue lost to own price impact rises with size). Ours is one learned battery against fixed bidders, not several competing stores.

## Self-cannibalisation of arbitrage revenue

4. Price cannibalisation, RAP deep dive. https://blueprint.raponline.org/deep-dive/price-cannibalisation/ (snippet; practitioner source)
5. "Should we be concerned about price cannibalisation of battery energy storage systems?" Ampersand Partners. https://www.ampersand.partners/post/should-we-be-concerned-about-price-cannibalisation-of-battery-energy-storage-systems (snippet; practitioner blog)
6. "Microeconomic models of electricity storage: Price forecasting, arbitrage limits, curtailment insurance, and transmission line utilization." Energy Economics, 2021. https://www.sciencedirect.com/science/article/abs/pii/S0140988321002899 (snippet)
   Practitioner and snippet sources agree on the qualitative claim: a large enough store is a price maker and compresses the spread it earns from. These are not evidence for our numbers; they are context for the cannibalisation panel.

## Learning-based storage and MARL market simulation

7. "Finding individual strategies for storage units in electricity market models using deep reinforcement learning." Energy Informatics (Springer), 2023. https://link.springer.com/article/10.1186/s42162-023-00293-0 (snippet; fetch hit a login redirect)
   Snippet says it uses MATD3 (a MADDPG variant with TD3 changes) to find strategies for storage units in market models. Closest method-side neighbour. Authors, market model and findings not verified.
8. "Multi-Agent Reinforcement Learning with Graph Convolutional Neural Networks for optimal Bidding Strategies of Generation Units in Electricity Markets." arXiv:2208.06242. https://arxiv.org/pdf/2208.06242 (snippet)
9. "Reinforcement Learning Based Bidding Framework with High-dimensional Bids in Power Markets." arXiv:2410.11180. https://arxiv.org/pdf/2410.11180 (snippet)
10. "Safe multi-agent deep reinforcement learning for joint bidding and maintenance scheduling of generation units." arXiv:2112.10459. https://arxiv.org/pdf/2112.10459 (snippet)
11. "Multi-agent deep reinforcement learning for efficient multi-timescale bidding of a hybrid power plant in day-ahead and real-time markets." Applied Energy, 2022 (ScienceDirect S0306261922004603). (snippet)
12. "Bidding a Battery on Electricity Markets and Minimizing Battery Aging Costs: A Reinforcement Learning Approach." Energies 15(14), 2022. https://doi.org/10.3390/en15144960 (snippet)
13. "Economic Battery Storage Dispatch with Deep Reinforcement Learning from Rule-Based Demonstrations." arXiv:2504.04326. (snippet)
   Items 8-13 are RL bidding or dispatch studies. Most treat the battery as a price taker or study one learner; none of the snippets mention measuring the market-wide effect of adding a battery against a same-seed baseline. That is the gap our paired design targets, but I have not read these papers, so "nobody does this" is not a claim I can make yet. Two caveats noted by the search summaries: MARL has no general convergence guarantee, which is why we pretrain generators and report convergence per seed.

## Empirical context (practitioner and operator sources)

14. CAISO, "2024 Special Report on Battery Storage" (29 May 2025). https://www.caiso.com/documents/2024-special-report-on-battery-storage-may-29-2025.pdf (snippet; not read)
15. ESS News, "Batteries reshape solar pricing in California market" (9 April 2026). https://www.ess-news.com/2026/04/09/batteries-reshape-solar-pricing-in-california-market/ (snippet)
   Search summaries say volatility fell in ERCOT and CAISO in 2025 and that battery charging raises mid-day prices (up to $42/MWh in CAISO for negatively priced solar). Trade-press claims; treat as pointers to primary data, not citations.
   Relevance: the direction of our peak-price and scarcity results (lower) is consistent with the volatility story; our mid-day/off-peak result (small rise for larger batteries) is consistent with charging demand lifting low prices. Our market is a stylised 3-generator system, so this is directional only.

## Not verified (from memory, not found in this search)

- Sioshansi, Denholm, Jenkin, Weiss and others on storage welfare and ownership structure around 2009-2011.
- Work on Nash equilibria of strategic bidding with storage (for example Ye, Papadaskalopoulos, Strbac and co-authors on price-making storage and deep RL bidding).
- Agent-based electricity market simulators such as AMIRIS and ASSUME.
  Search these by name before citing; I could not confirm titles, years or authors here.

## What this suggests for the write-up

- Frame the contribution as an empirical simulation of distributional effects with a paired, same-seed design, and cite Sioshansi (2014) as the theory predicting that sign pattern.
- Cite Cruise et al. for self-cannibalisation, noting ours is a single learner.
- Say plainly that the RL-bidding literature is large and that I have not established novelty beyond the paired market-level measurement.
