# Abstract

*Draft, 2026-10-11 — all sweeps have run, §5 has numbers. Still worth a close human read before
submission: an abstract compresses a lot, and compression is where overclaiming hides.*

We study whether an ant-colony metaheuristic over the federated-learning aggregation weight
vector — searching a per-round reweighting of client updates, with pheromone persisting across
rounds as a record of client reliability — improves on fixed and learned weighting for
non-IID, multi-site brain-tumour MRI classification. We report four findings. First, the
natural surrogate fitness for this search admits a degenerate optimum that standard health
signals do not reveal: the best single-client vertex outscores the FedAvg reference point in
every round we measured (15/15, mean margin +0.63), and we give a closed-form condition on the
penalty term that removes it. Second, at equal evaluation budget the colony does not separate
from simpler search — not on a synthetic heterogeneous fixture, where coordinate-grid search,
particle swarm optimization, a genetic algorithm and uniform random search all gain more, and
not on real data, where *none* of the five methods, ACO included, beats the FedAvg point under
heterogeneity. Third, cross-round pheromone persistence — the one mechanism here with no
analogue in the closest published prior work on swarm-optimized aggregation weights — does pay
on real data: full persistence outperforms both no persistence and the method's own default in
both heterogeneous regimes tested. Fourth, the server-side search cost is negligible (2.6% of
round wall-clock at K=10), and we report the measured constant rather than fitting the
unobserved O(K²) shape to it. Methodologically, we contribute a reusable diagnostic for
degenerate objectives in search-based aggregation weighting (`corner_margin`) with its
closed-form remedy, the first equal-budget comparison across five search methods for this
problem on both synthetic and real data, the first real-data test of whether cross-round state
helps it, and a component-level, pseudo-patient data split that catches chain-linked leakage an
image-level de-duplication pass (already published on this archive) leaves behind. Experiments
cover one dataset, K=10 clients, three partition regimes, 6 converged baselines (a seventh,
SCAFFOLD, was found non-convergent on this setup and is reported but excluded from headline
comparisons — §5.1), and 8 seeds where a significance claim is made. We argue the diagnostic,
the protocol, and the persistence result are the paper's contribution — not a claim that this
search method improves federated aggregation.

---

## Checklist before this is final

- [x] Confirmed, 2026-10-12: robustness raw JSONs and Figures 1-2 landed and don't change this
      abstract's numbers. SCAFFOLD's fix was never re-run on real Kaggle hardware (`HANDOVER.md`
      §5 item 1, closed without it) and this draft closes on that basis — "6 converged baselines
      (a seventh, SCAFFOLD, ... excluded)" stays as written, not a placeholder for 7.
- [ ] A human, not an automated search, should re-read the four findings against the full
      `05_RESULTS.md` and confirm none of them overstate what a first read of the tables
      supports — particularly finding 2 (real-data equal-budget) and finding 3 (A2 at 5 seeds).
- [ ] Word count / venue formatting — not checked here, this is content only.
