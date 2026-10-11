# LinkedIn post draft (to post manually)

I wanted to know what happens to a power market when a battery shows up, and who pays for it.

So I built a small simulation: three generators that learn to bid by reinforcement learning, trained first until their bidding settled. Then I dropped in a learning battery (5 to 75 MW, 1 to 8 hours of storage) and compared the same market, same demand, same seed, with and without it. 20 battery sizes, 10 seeds each, every one a paired comparison.

What came out (all numbers relative, because the simulator's prices and money are made up, so I only report percentages and per-MW yields):

- Prices fall a little. The average price drops by at most 1.7% against the same market without the battery, and the consumer bill by at most 2.8%. Small batteries (5 to 10 MW) barely move either one.
- Generators are the ones who pay. Their profit falls in every one of the 20 configurations, by 1.3 to 8.4%, and they lose 1 to 2.7 times what the battery earns. The savings come out of generator margins, not out of thin air.
- Bigger is not better for the battery. Revenue per MW falls from a yield of about 9 to 12% at 5 MW to 2 to 4% at 50 and 75 MW, because it eats its own spread.
- It does not necessarily shave the peak. In 12 of 20 configurations the net demand peak went up, because the battery charges into it.

Caveats, because they matter: the generators are frozen in this run, so it shows the short-run effect before anyone adapts, not the long-run equilibrium. The battery learns from a single training run, and 73 of the 200 trained batteries cycled less than 0.6 times a day, mostly the large and long ones. In a spot check on four sizes, an optimiser that searches the battery's schedule directly found about the same revenue and cycling as the learner, so the low cycling looks like what the market pays for once the battery moves its own price, not just a training problem. I have only checked four cases so far.

The theory has been around for a while (Sioshansi's work on storage and welfare predicts this winners-and-losers pattern). What I wanted was a paired simulation that measures it end to end. Write-up and code are on my GitHub; I am turning this into a proper paper.

If you work on storage trading or market design, I would like to hear what you would test next.

#energystorage #batteries #electricitymarkets #reinforcementlearning

---

Notes for you, not for the post:
- Every number above comes from outputs/bess_sweep/full (report.md paired-effects table, plus the ratio of generator loss to battery profit, which I computed from results.csv per configuration (summed over seeds)). Yield per MW is the percent of what 1 MW sold flat all day would earn at the no-battery mean price. Re-check after any rerun.
- Add the github.io link where marked once the page is live. The code is on the fork, github.com/eugkim0120/MARL, branch main; results are under results/bess_full/.
- Cut the caveat paragraph shorter if you want, but I would keep the "frozen generators" and "73 of 200 lightly used" points in.
