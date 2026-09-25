# References — resolved against the literature, with what is still open

Each entry below was retrieved from a live literature search (Consensus / Semantic Scholar
index) on 2026-09-25 and the metadata is quoted as returned. **Nothing here is reconstructed
from memory.** Entries still marked OPEN could not be resolved and must be before submission.

## Resolved

### FedAAW — analytically derived aggregation weights
> Xiaodong Li et al. (2025). *Federated Learning With Adaptive Aggregation Weights for Non-IID
> Data in Edge Networks.* IEEE Transactions on Cognitive Communications and Networking.
> 22 citations.
> https://consensus.app/papers/details/fe72b05ef602535a9667152c1dd2c5c9/

Matches plan §1.2's description exactly: "we theoretically derive an analytical expression for
the aggregation weights by minimizing the convergence upper bound of standard FL on non-IID
data across nodes". Positioned as a drop-in weight strategy with negligible communication
overhead.

⚠️ **Name collision — check which one you mean.** A second, different paper also calls its
method FedAAW: Qiaoyun Yin et al. (2024), *Tackling data-heterogeneity variations in federated
learning via adaptive aggregate weights*, Knowledge-Based Systems, 7 citations
(https://consensus.app/papers/details/9ca685ffaddd5ff187b581516444650b/). That one updates
weights using other clients' gradient information to keep weight updates consistent with the
global objective. The plan's "minimize a convergence upper bound" description fits Li et al.
Cite whichever is meant, and disambiguate if both.

### FedLAW — learned aggregation weights
> Zexi Li et al. (2023). *Revisiting Weighted Aggregation in Federated Learning with Neural
> Networks.* 146 citations.
> https://consensus.app/papers/details/e3b1614377c75e02bae69befe98cdc9a/

Confirmed as the origin of the name: "we propose an effective method for Federated Learning
with Learnable Aggregation Weights, named as FedLAW." Two findings worth using in §2.1 rather
than only citing: the aggregation weights need not sum to 1 (a "global weight shrinking" effect
that improves generalization), and client *coherence* governs which clients matter, with a
critical point during training before which more coherent clients dominate generalization.

The coherence result is directly relevant to this project's alignment term and should be
discussed, not just listed — it is prior work on the same signal.

## ⚠️ Closest competitor, found while resolving citations — must be addressed, not just cited

> Changlong Shi et al. (2025). *FedAWA: Adaptive Optimization of Aggregation Weights in
> Federated Learning Using Client Vectors.* CVPR 2025. 49 citations.
> https://consensus.app/papers/details/8069596a8a7c5b93b57bf2977f852d0a/

**This occupies the ground §2.1 claims is open.** From the abstract: it "adaptively adjusts
aggregation weights based on client vectors during the learning process. The client vector
captures the direction of model updates, reflecting local data variations, and is used to
optimize the aggregation weight **without requiring additional datasets or violating privacy**.
By assigning higher aggregation weights to local models whose updates align closely with the
global optimization direction..."

Our §2.1 differentiates this work from prior art on four axes: no server-side data, no
convexity/dissimilarity bound, more than one dimension of client signal, no cross-deployment
policy training. **FedAWA satisfies the first two and arguably the third**, and its weighting
signal — alignment of a client's update direction with the global direction — is the same
quantity as our fitness function's alignment term.

What remains distinct is narrower than the plan assumed, and honest statements of it are:

1. FedAWA optimizes $\alpha$ by gradient descent on client vectors; we search it with a
   population metaheuristic over a discretized level set.
2. FedAWA is stateless across rounds; cross-round pheromone persistence has no analogue in it.
   Ablation A2 measures whether that persistence is worth anything — which makes A2 the
   experiment carrying the novelty claim, not a secondary ablation.
3. Our fitness combines alignment with a dispersion term and a concentration penalty rather
   than alignment alone.

⚠️ **Action before submission:** §1.2 and §2.1 must be rewritten to cite FedAWA as the closest
prior work and state the distinction in those terms. A reviewer who knows CVPR 2025 will find it
immediately, and "applying a metaheuristic to the aggregation weight vector is the open slot"
does not survive contact with it as currently phrased. This also raises FedAWA to a *baseline*
worth running if hours allow — it is closer than FedLAW, which is currently the load-bearing
comparison.

### Krum — Byzantine-robust aggregation
> Peva Blanchard et al. (2017). *Machine Learning with Adversaries: Byzantine Tolerant Gradient
> Descent.* 3,028 citations.
> https://consensus.app/papers/details/b467033b38035bf59cf3f28e7354850c/

The canonical reference. Note for §5.3: Krum's time complexity is $O(n^2(d + \log n))$, which is
worth stating next to our own $O(K^2 d)$ Gram precompute — the robust baseline we compare against
has the same quadratic client dependence we are asked to justify.

### An attack our R2 arm is an instance of
> Cong Xie et al. (2019). *Fall of Empires: Breaking Byzantine-tolerant SGD by Inner Product
> Manipulation.* 369 citations.
> https://consensus.app/papers/details/b705e5ec78b35d86980f015b8e5f962e/

Shows coordinate-wise median and Krum can both be broken by attacks that manipulate the inner
product rather than the magnitude. **Our `sign_flip` attack preserves the update norm and
reverses its direction — an inner-product manipulation.** So R2's sign-flip arm is a known-hard
case for the very baselines it compares against, and §5.3 should cite this rather than present
the result as novel difficulty.

### Trimmed mean
> Tianxiang Wang et al. (2025). *Federated learning framework based on trimmed mean aggregation
> rules.* Expert Systems with Applications. 113 citations.
> https://consensus.app/papers/details/efa724340ac65a43bc94a6e809397182/

## ⚠️⚠️ The novelty claim is refuted — swarm-optimized aggregation weights already exist

§2.2 asserted that swarm methods in FL operate only at the systems layer and none optimizes the
aggregation weight vector. **That is false.**

> T. Srinivas et al. (2025). *Adaptive Federated Learning through Particle Swarm Optimization for
> Dynamic Client Weighting* (Adp-FL-PSO). NMITCON 2025. 0 citations.
> https://consensus.app/papers/details/e3e4425be03a568bb2942997c36d836e/

From the abstract: "an enhanced Particle Swarm Optimization (PSO) algorithm is applied to **the
server to compute the optimal aggregation weights**, thereby improving the model accuracy,
convergence, and data privacy." That is a swarm metaheuristic searching $\alpha$ server-side —
the exact slot this project was built to occupy, with PSO in place of ACO.

> Sunghwan Park et al. (2021). *FedPSO: Federated Learning Using Particle Swarm Optimization to
> Reduce Communication Costs.* Sensors. 119 citations.
> https://consensus.app/papers/details/e086645453185a4f8d34b9e39085a69a/

Replaces FedAvg's weight aggregation with PSO, though its stated goal is communication cost
rather than heterogeneity.

**This is bad news for the original framing and good news for the paper we can actually write.**
Our own A1 measures PSO at **+0.0484** against ACO's **+0.0013–0.0066** on identical fitness and
budget. The literature converged on PSO for this job; our controlled comparison independently
finds PSO beating ACO on it. Those agree. The honest contribution is therefore not "we applied a
swarm method to aggregation weights" — that exists — but:

1. the first **equal-budget, equal-fitness** comparison across ACO, PSO, GA, coordinate grid and
   random search for aggregation-weight optimization, which is what tells you *which* search
   matters rather than that search helps;
2. the degeneracy diagnostic (`corner_margin`) and its closed-form penalty threshold, which
   applies to **any** of these methods including the published PSO ones;
3. cross-round pheromone persistence, which none of FedAWA, Adp-FL-PSO or FedPSO has — still the
   one structural novelty, and still resting entirely on ablation A2.

⚠️ **Adp-FL-PSO should become a baseline**, not just a citation. It is the direct competitor.

### The systems-layer swarm citations, now resolved
> Gaith Rjoub et al. (2025). *A Hybrid Swarm Intelligence Approach for Optimizing Multimodal LLM
> Deployment in Edge-Cloud-based Federated Learning Environments.* Computer Communications.
> 18 citations. https://consensus.app/papers/details/3a8a1bf1af025bcbb10807d331eae6de/
> — PSO for selecting edge devices, **ACO for optimizing transmission of model updates**. Covers
> both the "ACO for transmission scheduling" and "PSO for device selection" markers.

> Koffka Khan et al. (2024). *Swarm Intelligence-Driven Client Selection for Federated Learning in
> Cybersecurity Applications.* 3 citations.
> https://consensus.app/papers/details/08858b14bb605524b58372376b099e34/
> — benchmarks nine SI algorithms including ACO and PSO for client selection. Useful: it finds
> Grey Wolf Optimization beating both, which is a second independent signal that ACO is not the
> strongest swarm choice in FL settings.

> Huayang Sun et al. (2025). *Federated multi-label feature selection via manifold sparse
> constraints and game-theoretic evolutionary ant colony optimization* (Fed-MGACO). Journal of
> King Saud University CIS. 4 citations.
> https://consensus.app/papers/details/38566e4442005e5db5aba243ba834a6f/
> — resolves the "ACO for federated feature selection" marker.

## OPEN — could not be resolved in this session

The literature search rate-limited before these could be retrieved. Each needs a real
reference; none may be filled from memory.

| marker | what it needs |
|---|---|
| FedNolowe | loss-based / heuristic aggregation weighting — confirm name, authors, venue, year |
| DaWa | RL-optimized aggregation weights — confirm name, authors, venue, year |
| FedProx | Li et al. — retrieve exact citation |
| SCAFFOLD | Karimireddy et al. — retrieve exact citation |
| client drift / non-IID convergence | the analysis §1.1 leans on |
| "FedACo" 2025 | the acronym-collision paper named in plan §0.3, to cite-and-differentiate |
| brain tumour MRI dataset | the canonical release and its originating publication |

Names in the right-hand column are the ones this project's own plan uses; treat them as search
terms, not as citations. Retrieve each and paste the returned metadata here verbatim, the way
the resolved entries above are recorded.

## ⚠️⚠️⚠️ The de-duplication contribution is ALSO not novel — corrected 2026-09-25

An earlier note in this session called the leakage/de-duplication audit "a contribution to the
data section" and the project's strongest asset. **That was asserted without checking the
literature, and it is wrong.** The audit has been done, on this exact dataset, at least three
times:

> Dong Lu et al. (2026). *MTA-Swin: A Multi-Token Attention Swin Transformer for Brain Tumor
> Classification with Leakage-Free MRI Benchmarking.* Journal of Medical Systems.
> https://consensus.app/papers/details/855f359e77b75fca8b0a0527d7a2c127/
> — "many widely adopted brain tumor MRI datasets suffer from duplicate-induced data leakage…
> we develop an automated data cleaning pipeline capable of identifying and removing duplicate
> scans… a **leakage-free benchmark dataset containing 3,522 unique MRI scans**."

> Md Ashik Khan et al. (2026). *WICA-Net-M …* Computers.
> https://consensus.app/papers/details/cda91b10631e589aa3e00aa968ce12f5/
> — evaluates on "the standard and **image-level deduplicated splits of the Nickparvar brain
> tumour MRI dataset** under a three-seed, leakage-aware protocol", reports a **4.17-point
> V1→V2 accuracy drop**, and audits 861 exact SHA-256 overlaps against BRISC. Explicitly names
> the confound we also found: the drop "reflects the combined effects of duplicate removal,
> **class rebalancing**, and altered sample composition".

> Mohiuddin Saifullah et al. (2025). *Reliable Brain Tumor Classification Without Metadata: A
> Step-by-Step Guideline with Duplicate Removal.* IEEE ICTAI 2025.
> https://consensus.app/papers/details/1fdc0eb9ec175f06b1c421925a964869/
> — "We apply **perceptual hashing (pHash)** to detect and remove duplicate or near-duplicate
> images, which can otherwise inflate performance", then compares before/after with nested CV.
> Same method, same problem, same dataset family.

Context that must also be cited:
> D. Wallis et al. (2022). *Clever Hans effect found in a widely used brain tumour MRI dataset.*
> Medical Image Analysis. 44 citations.
> https://consensus.app/papers/details/59cd77ed2fb35acab6dce69065ad511a/
> I. Tampu et al. (2022). *Inflation of test accuracy due to data leakage in deep learning-based
> classification of OCT images.* Scientific Data. 108 citations.
> https://consensus.app/papers/details/62a7d42741a75b3a8be900a4de1a6492/

### What is left of our data contribution — smaller, but real and not yet claimed

**Both 2026 papers de-duplicate at the IMAGE level and both state patient-level leakage as an
open problem they cannot solve.** Khan et al.: *"Patient-level leakage remains unresolved because
patient identifiers are unavailable."* Sabuj et al. (NeuroAttnFuseNet, 2026) say the same.

Our split does not use patient identifiers either — it builds a duplicate graph, takes connected
components, and treats **each component as one pseudo-patient**, then splits at that level so no
component can span train/val/test or two federated clients. That is a *proposed answer to the
open problem those papers state*, and none of the three implements it. §4.2 should be reframed
from "we audited this dataset" (done, by others) to "image-level de-duplication is insufficient;
here is component-level pseudo-patient splitting, and here is what it costs".

Also apparently unclaimed, and cheap to state precisely:
- **Per-class redundancy.** Khan et al. name class rebalancing as one of three inseparable
  confounds. We separate it: `notumor` is **67.9%** redundant at 3.120 images per pseudo-patient
  against 18.0–26.3% and 1.220–1.357 for the tumour classes, so the archive's exact 1,800-per-class
  balance is manufactured by duplicating one class, and de-duplication leaves 2.56:1 imbalance.
- **Source provenance.** 1,829/7,200 images (25.4%) hash-matched to a Figshare source, median
  match distance 0. Partial (SARTAJ and Br35H pending) but nobody else reports it.

### Consequence for the paper

Two of the three things this project counted as contributions are now gone: the ACO claim (fails
at equal budget) and the de-duplicated benchmark (published, three times). What remains:

1. **`corner_margin` and its closed-form penalty threshold** — a diagnostic for a degenerate
   optimum in search-based aggregation-weight objectives, applicable to the published PSO methods.
   Nothing found resembling it. **Strongest remaining card.**
2. **The equal-budget, equal-fitness protocol** across ACO/PSO/GA/grid/random. Each existing paper
   validates one method against FedAvg; none isolates which search matters.
3. **Cross-round stigmergy** — untested. A2 decides it.
4. **Component-level pseudo-patient splitting** — an answer to a stated open problem, worth a
   subsection rather than a contribution bullet.
