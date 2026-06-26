# Thesis Methodology: Scientific Method and Reproducibility Appendix

This document separates the scientific methodology from software-operation details. It is based on the repository state inspected through 25 June 2026. The authoritative status and artefact map are in `THESIS_HANDOFF_FOR_NEXT_MODEL.md`; when older text here conflicts with that handoff or a generated implementation audit, the handoff and audit take precedence. Values that are not supported by code or retained outputs are explicitly marked **needs confirmation**.

## 1. Methodological overview

### Phase 1: reproduction of published peptide classifiers

**Goal.** Reproduce the five sequence classifiers described by Njirjak et al. (2024), verify the dataset and preprocessing contract, select architecture hyperparameters without using an outer test fold, and freeze a common architecture for later adaptive experiments.

**Input.** A labelled experimental dataset of 368 unique peptide sequences. Labels denote experimentally reported self-assembling (`1`) and non-self-assembling (`0`) peptides.

**Method.** Five model families were evaluated: AP, SP, AP_SP, TSNE_SP, and TSNE_AP_SP. Stratified nested cross-validation used five outer folds and five inner folds. For each outer fold, the inner folds selected the number of recurrent cells and convolutional kernel size by minimum average inner-validation loss. The untouched outer fold estimated generalization. Threshold-dependent metrics were reported at 0.5 and at thresholds selected from inner-validation predictions.

**Outputs.** Dataset checks, fold assignments, inner-fold tuning results, outer-fold predictions, selected thresholds, frozen architecture settings, and reproduced predictive-performance tables.

**Supported claim.** The reproduced AP_SP model closely recovered the performance reported by Njirjak et al.; the reproduced PR-threshold aggregate was accuracy 0.8180 and F1 0.8686.

**Unsupported claim.** Phase 1 does not demonstrate prospective discovery or improvement through active learning.

### Phase 2: retrospective active-learning replay

**Goal.** Compare acquisition strategies under controlled conditions where labels are known but hidden from the selection algorithm.

**Input.** For each outer fold, the outer test fold was treated as holdout, one inner fold as validation, and the remaining 235 peptides as the replay training pool. Initial labelled subsets of 10 and 40 peptides were sampled stratifiably from that pool; all remaining training-pool labels were hidden.

**Method.** At each round, models were retrained from scratch on the currently revealed labels. The hidden training pool was scored, five peptides were selected, their already-known experimental labels were revealed, and the process continued for at most 20 acquisition rounds. Validation selected the \(F_1\)-maximising decision threshold with higher-threshold tie-breaking; that threshold was frozen and applied unchanged to the holdout only for threshold-dependent reporting. Ten acquisition strategies were compared in the main benchmark: random, ensemble mean, similarity-penalised mean, predictive entropy, ensemble mutual information, UCB, family QBC, cluster-diverse representative selection, OED logdet, and hybrid MI-diverse. PI, EI, and MES were not main Phase 2 replay strategies. An ablation compared single versus five-member ensembles and raw versus calibrated probabilities.

The official replay evaluates \(n_0=10\) and \(n_0=40\) separately, with
batch size \(B=5\) and at most 20 acquisition rounds. Labels-to-target are
reported at \(F_1=0.80,0.84,0.86\); \(0.84\) is the principal practical
target. Targets are detected only at observed replay points, without
interpolation. Unreached targets remain not reached, and reach counts or reach
fractions accompany successful-fold labels-to-target summaries.

**Outputs.** Per-round validation and holdout metrics, selected sequences, complete acquisition logs, AULC-F1, labels-to-target summaries, calibration summaries, overlap matrices, diversity summaries, and presentation-ready figures.

**Supported claim.** Phase 2 measures retrospective label efficiency and strategy behaviour on the published experimental dataset.

**Unsupported claim.** Phase 2 does not generate new peptides, run CG-MD, or provide prospective experimental labels.

### Phase 3: real branch-isolated active learning

**Goal.** Prospectively generate peptides, evaluate them by CG-MD, review evidence manually, ingest operational computational labels, and update strategy-specific models over multiple rounds.

**Input.** Each strategy branch began with the full 235-peptide experimental training pool from outer fold 1/inner fold 1. The fixed validation set contained 59 peptides and the frozen holdout contained 74 peptides. The `replay_seed_size=40` setting remains in configuration for replay compatibility but does **not** define the Phase 3 training set.

**Selected branches.** Predictive entropy, family QBC, and cluster-diverse representative. Ensemble mutual information was retained as a backup.

**Method.** Each branch maintained an independent ledger. Before a proposal round, a fresh model ensemble was trained from that branch's current ledger using only the frozen Phase 1 architecture. A genetic algorithm generated at least 50 unique candidates. The branch acquisition rule selected five peptides. Each selected peptide was simulated by CG-MD, evidence was reviewed by a human, and a branch-local label file was created. Ingestion updated only the selected branch ledger and did not retrain automatically. The next proposal retrained from scratch from the updated ledger. A new round was blocked unless every row in the previous `selected_batch.csv` was acquired with a valid label.

**Terminal protocol.** The study stops after Round 8. After Round 8 ingestion, a separate finalization job retrains each final model on 235 experimental labels plus 40 branch-specific CG-MD labels, giving 275 labelled peptides per branch if all eight batches contain five successfully ingested rows. It then evaluates validation and, once, the frozen holdout. No Round 9 candidates are generated.

**Supported claim.** Branch-specific CG-MD feedback changes the corresponding branch training data and allows comparison of three prospective acquisition trajectories.

**Completed evidence.** All three branches completed eight five-peptide rounds, terminal retraining on 275 labels, and frozen-holdout evaluation. Final holdout F1 was 0.8750 for cluster-diverse representative selection, 0.8713 for family QBC, and 0.8762 for predictive entropy. These results support descriptive branch comparison, not universal strategy superiority or independent external validation. The canonical completed archive is `thesis_results/03_real_al/phase3_results_20260620/thesis_results/03_real_al/`.

### Phase 4: single-round BO-guided peptide proposal

**Goal.** Optimize a fixed trained surrogate's candidate shortlist rather than improve the surrogate itself.

**Input.** A branch-neutral fixed split recreated from Phase 1: 235 experimental peptides for model fitting, 59 for fixed validation/calibration, and 74 frozen-holdout sequences whose labels and predictions remain unavailable. Holdout identities are used only for exact duplicate exclusion.

**Method.** Five independently seeded `AP_SP` members are fitted on the 235-row partition and calibrated member-wise on the fixed 59-row validation/calibration partition before aggregation. The same immutable checkpoints, calibrators, calibrated training predictions, and surrogate-space incumbent are shared by all guided policies. Separate candidate pools are generated for random, calibrated greedy, calibrated UCB, probability-space approximate PI, probability-space approximate EI, and calibrated ensemble-based approximate MES. Five candidates are exported by each successful policy. The operation is single-round and does not ingest labels or create another acquisition round.

**Outputs.** The primary experiment was executed and aggregated: all six policies completed, each retained 50 candidates and selected five, and the comparison contained 30 exact-unique selected sequences. Canonical proposal results are under `thesis_results/04_bayesian_optimization/Phase4 results/phase4_complete_20260621/thesis_results/04_bayesian_optimization/`. Primary Phase 4 CG-MD evidence has also been reconciled for 29 valid proposals; 15 satisfy the operational threshold conjunction `AP_sasa(200 ns) >= 1.75 AND paper_path_APcontact_last10ns >= 0.5`. One retained MES selection has length 25, outside the declared 3--24 range, and was not simulated; this historical artefact must be disclosed rather than silently rewritten.

**Supported claim.** The implementation compares BO-style utilities for candidate optimization under a fixed surrogate.

**Unsupported claim.** It is not a closed-loop Bayesian-optimization campaign and does not establish experimental superiority of a BO acquisition function.

### Phase 5: self-paced learning

**Status: executed and aggregated.**

Phase 5 is an isolated retrospective replay with one calibrated `AP_SP` model
per replay point. It compares random, predictive entropy, static easy-entropy,
and progressively self-paced entropy from 10 to all 235 replay-training labels
over three overlapping outer folds. Difficulty is an operational calibrated
neural-model labelled-manifold familiarity proxy, not a chemical-complexity or
experimental-cost score. All 12 jobs and aggregation completed. Predictive
entropy produced the highest mean full AULC-F1 (0.8223), followed by random
(0.8171), self-paced entropy (0.8147), and static easy entropy (0.8016). Full
details are given in Section 16 and `PHASE5_RESULTS_SUMMARY.md`.

## 2. Dataset and label definition

The dataset is distributed with the implementation of Njirjak et al. (2024), *Reshaping the discovery of self-assembling peptides with generative AI guided by hybrid deep learning*, Nature Machine Intelligence, DOI `10.1038/s42256-024-00928-1`.

| Property | Verified value |
|---|---:|
| Total peptides | 368 |
| Experimental self-assembling (`1`) | 249 |
| Experimental non-self-assembling (`0`) | 119 |
| Positive fraction | 67.663% |
| Negative fraction | 32.337% |
| Minimum length | 3 |
| Maximum length | 24 |
| Alphabet | `ACDEFGHIKLMNPQRSTVWY` |
| Duplicate sequences | 0 |
| Outer-fold-1 training pool | 235: 159 positive, 76 negative |
| Outer-fold-1 validation | 59: 40 positive, 19 negative |
| Outer-fold-1 holdout | 74: 50 positive, 24 negative |

Rows with labels other than `0` or `1` are ignored. Duplicate sequences cause a hard validation error. Splits are stratified by class. Phase 1 uses nested 5-by-5 stratified cross-validation with shuffling. Phase 2 uses outer folds 1–5 and inner fold 1, with deterministic run seeds

\[
s=s_0+1000\,k_{\mathrm{outer}}+100\,k_{\mathrm{inner}}+n_0,
\]

where \(s_0=20260317\) and \(n_0\in\{10,40\}\).

Class imbalance is handled by weighting negative samples:

\[
w_0=\frac{N_+}{N_-},\qquad w_1=1.
\]

For the complete dataset this corresponds approximately to \(w_0=249/119=2.092\). The weight is recalculated from the training rows in each adaptive round.

Experimental labels represent the source dataset's reported assembly class. Phase 3 CG-MD labels are separate **operational computational labels**, not universal experimental ground truth.

## 3. Feature representations

All sequences are truncated/accepted only up to the verified maximum length 24. Numerical inputs are scaled to \([-1,1]\); padding/masking uses value `2`, which lies outside that range.

### AP

Three sequence-aligned channels are constructed from published aggregation-propensity lookup maps:

1. single-amino-acid AP values;
2. overlapping dipeptide AP values;
3. overlapping tripeptide AP values.

Each lookup dictionary is min-max scaled to \([-1,1]\), and each sequence is padded to length 24 for AP-only models.

### SP

`seqprops.SequentialPropertiesEncoder` generates 95 physicochemical-property channels. The first channel is disabled, leaving 94 channels. Values are min-max scaled to \([-1,1]\). The non-TSNE SP representation is padded to length 25 in the current implementation.

### TSNE-derived SP

Three precomputed amino-acid lookup maps are loaded from one-, two-, and three-component TSNE files. A sequence is mapped residue by residue through each map and padded to length 24. These are fixed precomputed representations; TSNE is not re-fitted inside each cross-validation fold.

| Model family | Input representation |
|---|---|
| AP | Three AP branches: amino acid, dipeptide, tripeptide |
| SP | One sequence tensor containing 94 physicochemical channels |
| AP_SP | Three AP branches plus one 94-channel SP tensor |
| TSNE_SP | Three TSNE-derived sequence channels |
| TSNE_AP_SP | Three AP branches plus a three-channel TSNE-derived tensor |

Exact runtime probing with the pinned TensorFlow 2.10.1 environment confirmed the following effective Keras inputs: AP uses three `(24,1)` branches; SP uses `(25,94)`; AP_SP uses three `(25,1)` AP branches followed by `(25,94)` SP; TSNE_SP uses `(3,24)` without transposition; and TSNE_AP_SP uses three `(24,1)` AP branches followed by a transposed `(24,3)` t-SNE branch. AP arrays are supplied by NumPy as `(B,24)` or `(B,25)` and Keras adds the singleton feature dimension. The standalone TSNE_SP orientation means Conv1D treats 3 as the temporal dimension and 24 as channels; whether that orientation was intended by the original authors **needs confirmation**.

## 4. Predictive model architecture

### AP model

Each of the three AP channels is processed independently by:

1. masking with mask value 2;
2. a bidirectional LSTM with 5 units in each direction and sequence output;
3. a unidirectional LSTM with 5 units;
4. a SELU dense layer with width \(2h\), where \(h\) is the selected architecture-size parameter;
5. dropout 0.5.

For the frozen AP configuration, \(h=64\), so each branch has a 128-unit dense layer. The recurrent widths nevertheless remain fixed at 5 units: the selected value 64 does not configure the AP LSTMs. The three 128-dimensional branch outputs are concatenated into a 384-dimensional vector and passed directly to one sigmoid output neuron.

### SP and TSNE_SP models

Each model receives one sequence tensor and applies masking with value 2, two same-padded Conv1D layers with five filters each and linear activation, one bidirectional LSTM, dropout 0.5, and a one-unit sigmoid output layer. The convolutional kernel size and number of LSTM units per direction are selected by the frozen configuration. SP uses kernel size 6 and 32 LSTM units per direction, producing a 64-dimensional recurrent output. TSNE_SP uses kernel size 6 and 48 units per direction, producing a 96-dimensional recurrent output.

The masking layer precedes Conv1D, but Keras Conv1D does not propagate a temporal mask. The architecture can therefore be described as containing a masking layer before convolution, but the downstream bidirectional LSTM should not be described as receiving a guaranteed intact Keras mask.

### AP_SP and TSNE_AP_SP models

Each hybrid model processes its first three AP inputs independently using the same fixed recurrent structure as AP: masking, a 5-unit-per-direction bidirectional LSTM with sequence output, a 5-unit unidirectional LSTM, a SELU dense layer, and dropout 0.5. The fourth SP or t-SNE branch is processed by masking, two five-filter Conv1D layers, one bidirectional LSTM, and dropout 0.5. The four branch outputs are concatenated and passed directly to a one-unit sigmoid output layer.

For AP_SP, \(h=48\): every AP branch has a 96-unit SELU layer, the SP branch uses 48 LSTM units per direction, and concatenation produces 384 features. Both Conv1D layers use kernel size 8. For TSNE_AP_SP, \(h=64\): every AP branch has a 128-unit SELU layer, the t-SNE branch uses 64 LSTM units per direction, and concatenation produces 512 features. Both Conv1D layers use kernel size 6.

### Frozen architectures

| Model | Selected \(h\) | AP recurrent units | Sequence-branch BiLSTM units per direction | AP dense width per branch | Conv1D kernel |
|---|---:|---:|---:|---:|---:|
| AP | 64 | BiLSTM 5 per direction, then LSTM 5 | not applicable | 128 | not applicable |
| SP | 32 | not applicable | 32 | not applicable | 6 |
| AP_SP | 48 | BiLSTM 5 per direction, then LSTM 5 | 48 | 96 | 8 |
| TSNE_SP | 48 | not applicable | 48 | not applicable | 6 |
| TSNE_AP_SP | 64 | BiLSTM 5 per direction, then LSTM 5 | 64 | 128 | 6 |

The adaptive phases use a five-member AP_SP ensemble. Members share the architecture but use different random seeds. AP_SP was chosen because it combines aggregation-propensity and physicochemical information, reproduced the published performance closely, and provides an ensemble distribution needed for uncertainty acquisition. The cost is approximately five training/inference passes, but the ensemble supplies mean probability, variance, entropy, and epistemic-disagreement estimates.

## 5. Training, validation, thresholding, and calibration

Models use Adam with initial learning rate \(0.01\), batch size 600, binary cross-entropy, 70 epochs, and dropout 0.5. The learning rate remains constant for 10 epochs and is then multiplied by \(\exp(-0.1)\) after every subsequent epoch. There is no patience-based early stopping. Instead, the weights with minimum validation loss are retained when validation data are available; otherwise minimum training loss is retained.

For sample \(i\), class-weighted binary cross-entropy is

\[
\mathcal L=-\frac{1}{N}\sum_i w_{y_i}
\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right].
\]

For an ensemble of \(M=5\) models,

\[
\bar p(x)=\frac{1}{M}\sum_{m=1}^M p_m(x).
\]

Each ensemble member is calibrated separately on validation predictions. The implementation applies Platt-style logistic calibration to the raw probability logit:

\[
z=\log\frac{p}{1-p},\qquad
p_{\mathrm{cal}}=\sigma\!\left(a\frac{z-\mu_z}{\sigma_z}+b\right).
\]

Parameters \(a,b\) are fitted by gradient descent for at most 500 iterations, learning rate 0.05, with L2 penalty \(10^{-3}\) toward the identity mapping \(a=1,b=0\). Structurally malformed calibration arrays fail explicitly. Valid degenerate cases use identity fallback when validation has one class, validation-logit population SD is below \(10^{-6}\), optimisation raises after valid inputs, or fitted parameters are non-finite. Identity fallback returns clipped raw probabilities unchanged.

The primary adaptive threshold is the validation threshold maximising F1 over observed probability values. Ties are resolved in favour of the higher threshold. In Phase 2, the validation-selected threshold is frozen and applied unchanged to the holdout. In final Phase 3 evaluation, the threshold is again selected only on validation and applied once to the frozen holdout. It is never selected on holdout predictions and is used only for threshold-dependent reporting.

## 6. Acquisition strategies

Let \(p_m(x)\) be calibrated probability from ensemble member \(m\), \(\bar p(x)\) the ensemble mean, and

\[
H(p)=-p\log p-(1-p)\log(1-p).
\]

| Strategy | Definition and selection |
|---|---|
| Random | Deterministic seeded shuffle; first \(B\) candidates |
| Ensemble mean | \(a(x)=\bar p(x)\); top \(B\) |
| Predictive entropy | \(a(x)=H(\bar p(x))\); top \(B\) |
| Ensemble MI | \(a(x)=H(\bar p)-M^{-1}\sum_m H(p_m)\); top \(B\) |
| UCB | \(a(x)=\bar p(x)+\beta\sigma_p(x)\), \(\beta=1\); top \(B\) |
| Family QBC | Five heterogeneous model families vote using calibrated \(p_m\ge0.5\); vote entropy is primary and probability SD breaks ties |
| Similarity-penalized mean | Greedy \(a(x)=\bar p(x)-S(x,R)\), updating references after each selected peptide |
| Cluster-diverse representative | Cluster embeddings, rank clusters by distance from labelled embeddings, select the member nearest each selected centroid |
| OED logdet | Greedy maximum log-determinant gain in embedding information matrix |
| Hybrid MI-diverse | Select \(\lceil B/2\rceil\) by MI, then fill by farthest-first embedding diversity |

The sequence-composition similarity penalty is

\[
S(x,R)=\frac{1}{|R|}\sum_{r\in R}
0.1\left(1-\frac{\lVert f(x)-f(r)\rVert_1}{|x|+|r|}\right),
\]

where \(f(x)\) is the vector of amino-acid counts. It is a composition similarity, not edit distance.

For family QBC, if \(v_m(x)=\mathbf 1[p_m(x)\ge0.5]\) and
\(q=M^{-1}\sum_m v_m\), then

\[
a_{\mathrm{QBC}}(x)=H(q).
\]

The 0.5 threshold is intentional for committee voting and is distinct from the validation-selected classification threshold.

For OED, with embedding \(e_x\), labelled embeddings \(E_L\), and
\(\lambda=10^{-3}\),

\[
A_0=\lambda I+E_L^\top E_L,\qquad
\Delta(x\mid A)=\log\det(A+e_xe_x^\top)-\log\det(A).
\]

The batch is selected greedily, updating \(A\) after every choice.

Cluster-diverse representative uses \(K=\min(N,\max(B,\lceil3B\rceil))\) k-means-like clusters. Cluster novelty is the Euclidean distance from a cluster centroid to the nearest labelled embedding. Clusters are sorted by decreasing novelty, then by representative distance to centroid, then input index. One centroid-nearest representative is selected per cluster.

Hybrid MI-diverse does not combine MI and diversity into one scalar. It forms a union of two subsets: the first by MI and the second by dynamic farthest-first distance from labelled and already selected uncertain points.

Stable descending sorts preserve input order for exact ties. QBC additionally uses probability SD; similarity selection uses unpenalized mean as a secondary key; cluster selection uses centroid distance and index.

## 7. Phase 2 retrospective replay

For every outer fold, experimental labels in the 235-peptide training pool were hidden except for a stratified initial set of 10 or 40. Validation and holdout labels remained available only for evaluation. Each acquisition round:

1. trained the required model(s) from scratch;
2. fitted calibration on validation;
3. selected a validation F1 threshold;
4. evaluated validation and holdout;
5. scored all hidden training-pool candidates;
6. selected five candidates;
7. revealed their experimental labels.

The process ran for up to 20 rounds. Five outer folds served as five fold-level repetitions with different stratified outer test partitions and deterministic initial seeds. Because their training partitions overlapped, these repetitions were not treated as statistically independent. Random and all other strategies were evaluated under the same five fold-specific conditions.

Normalized AULC-F1 is

\[
\mathrm{AULC}_{F_1}
=
\frac{1}{L_T-L_0}
\sum_{t=0}^{T-1}
\frac{F_{1,t}+F_{1,t+1}}{2}
\left(L_{t+1}-L_t\right).
\]

This is the exact implemented discrete trapezoidal calculation over the observed labelled-count points. AULC values for \(n_0=10\) and \(n_0=40\) are calculated and reported separately because they cover different labelled-count intervals. Labels-to-target is the first observed labelled count at which F1 reaches a target (0.80, 0.84, or 0.86); no interpolation is performed. Terminal F1 is the F1 at the final executed replay point.

Main combined holdout AULC-F1 values used for Phase 3 selection were predictive entropy 0.8281, ensemble MI 0.8203, family QBC 0.8197, cluster-diverse 0.8149, and random 0.8086. These are aggregate retrospective results, not prospective CG-MD outcomes.

## 8. Phase 3 real branch-isolated active learning

The three main strategies were selected by a composite evidence procedure:

\[
0.35\,s_{\mathrm{AULC}}+
0.25\,s_{\mathrm{labels-to-target}}+
0.15\,s_{\mathrm{rank}}+
0.15\,s_{\mathrm{diversity}}+
0.10\,s_{\mathrm{nonredundancy}}+
\text{role bonus}.
\]

A strategy also had to clear the random AULC floor. The final trio was role-diverse rather than simply the top three AULC values: predictive entropy represented decision uncertainty and best replay efficiency; family QBC represented disagreement between model families; cluster-diverse representative provided a deliberately non-redundant CG-MD slate.

Each branch began with the same 235 experimental labels. Branches share frozen validation and holdout sets but do not share acquired CG-MD labels. The shared MD inventory records provenance only. A sequence selected independently by two branches remains visible in both branches and is not automatically copied or merged.

Candidate generation uses the branch's acquisition-matched GA objective. Similarity and length penalties shape GA evolution in the actual Phase 3 configuration. Final acquisition then applies the branch-specific batch rule to the generated pool. Five selected peptides per branch per round are sent to CG-MD.

Ingestion validates branch, round, membership in that round's `selected_batch.csv`, labels, and duplicates. It preserves acquisition score, rank, selected-batch path, source round, initial-seed/replay provenance, and label-source history. Partial ingestion sets `partially_ingested` and blocks continuation.

The completed design contains eight rounds, hence 40 CG-MD acquisitions per branch. Final validation is available round by round; the frozen holdout is evaluated only after terminal Round 8 retraining to avoid using the holdout as a repeated decision tool.

## 9. Genetic-algorithm peptide proposal

The GA uses the 20 canonical amino acids. Initial sequence length is uniformly sampled from 3–24 residues. Preferred output length is 5–10; outside that interval the penalty is

\[
P_L(x)=\min\!\left(0.05\left||x|-7.5\right|,0.5\right).
\]

Configuration:

| Parameter | Value |
|---|---:|
| Population | 50 |
| Offspring per generation | 30 |
| Generations | 30 |
| Tournament size | 3 plus initial random contender |
| Mutation probability per child | 0.05 |
| Candidate target | 50 unique peptides |
| Phase 3 independent attempts | up to 100 |

Parents are selected by tournament fitness. One-point proportional crossover joins a prefix of parent 1 with a suffix of parent 2. If mutation occurs, one of four operations is chosen: insertion, swap, deletion, or substitution. Parents and offspring are rescored together, and the top 50 by the complete generation fitness survive.

For Phase 3,

\[
F_{\mathrm{GA}}(x)=U_{\mathrm{strategy}}(x)-S_{\mathrm{population}}(x)-P_L(x),
\]

where \(U_{\mathrm{strategy}}\) is mean probability, entropy, MI, QBC vote entropy, embedding novelty, OED single-point gain, or a BO utility, depending on the branch. Random uses a broad-pool objective. Cluster-diverse uses normalized distance to the nearest labelled embedding during generation. Hybrid generates separate MI and embedding-novelty subpools, merges and deduplicates them, and applies the hybrid final selector afterward.

Generated sequences already present in the ledger are discarded. Duplicate generated sequences are deduplicated. Canonical alphabet and length rules prevent invalid sequences. After each GA attempt, novel candidates are ranked by the unpenalized generator utility for pool export; final batch selection is then performed by the acquisition rule.

Within the shared generative algorithm and retained-pool selection path, Phase
4 changes only the calculation of \(U_{\mathrm{policy}}(x)\); all Phase 3
penalty functions, genetic-algorithm mechanics, pool construction, and selector
behaviour remain unchanged.

## 10. Phase 4 BO-guided proposal

Let \(f^\star\) be the maximum predicted mean among currently labelled training peptides, \(\mu(x)=\bar p(x)\), \(\sigma(x)\) ensemble standard deviation, and \(\xi=0\).

Implemented acquisitions are:

\[
\mathrm{UCB}(x)=\mu(x)+\beta\sigma(x),\quad \beta=1;
\]

\[
z=\frac{\mu-f^\star-\xi}{\max(\sigma,10^{-8})},
\quad \mathrm{PI}(x)=\Phi(z);
\]

\[
\mathrm{EI}(x)=(\mu-f^\star-\xi)\Phi(z)+\sigma\phi(z).
\]

Phase 4 MES is a finite-pool calibrated neural-ensemble approximation. Each of
the five calibrated ensemble-member prediction vectors is treated as one
coherent approximate function sample. Temporary member-wise maxima may be
calculated over the current GA scoring population during evolution. Final
ranking recomputes one common set of five member-wise maxima over the complete
retained pool. It is not exact Gaussian-process MES. Thompson sampling is
**not implemented**.

Each policy generates a strategy-specific pool, rescoring and exporting five
candidates on successful completion. Phase 4 initialization writes the fixed
configuration, implementation audit, schemas, and SUPEK PBS workflow without
training or submission. No labels are ingested automatically.

The legacy `run-discovery` command is generic discovery support. It may load a
latest post-ingest model or baseline fallback and may use
`discovery_mes_samples=128`, but it is separate from the thesis Phase 4
experiment. Its outputs must not be used as Phase 4 thesis evidence.

## 11. Phase 5 self-paced training

The completed Phase 5 replay provides an explicit operational familiarity
proxy, a preregistered easy-to-hard eligibility schedule, matched random and
ordinary predictive-entropy controls, repeated retraining, and fold-matched
holdout evaluation. It remains SPAL-inspired rather than an exact
reimplementation of the paper's kernel optimisation. The completed result does
not show that self-paced acquisition improves overall AULC over ordinary
predictive entropy or random selection. It does show an improvement over the
permanently restricted static easy-only policy.

## 12. CG-MD evaluation and operational label

The full BURA protocol uses Martini 2.2P (`martini22p`) with polarized-water residue `PW`. A peptide is converted to an extended coarse-grained structure, inserted into a \(20\times20\times20\ \mathrm{nm}^3\) box, minimized in vacuum, solvated, minimized, equilibrated with Berendsen and then Parrinello–Rahman pressure coupling, and simulated for 200 ns.

The number of peptide copies is sequence-dependent:

\[
N_{\mathrm{mol}}=\left\lfloor\frac{1200}{|x|}\right\rfloor.
\]

Production uses timestep 0.020 ps, temperature 303 K, isotropic pressure 1 bar, v-rescale thermostat, Parrinello–Rahman barostat, reaction-field electrostatics, and 1.1 nm Coulomb/van der Waals cutoffs. One trajectory per selected peptide was run; independent MD replicas were not part of the Phase 3 operational protocol.

### 12.1 Exact full-profile simulation contract

The exact parameter contract is transcribed in
`THESIS_CGMD_PARAMETER_CONTRACT.md`. The principal implemented settings are:

- GROMACS 2023.2, mixed-precision MPI/OpenMP build;
- `martinize.py` 2.6 with Martini 2.2P (`martini22p`) and the refined
  polarizable-water topology `martini_v2.2refP.itp`;
- three steepest-descent stages with force thresholds 20, 100, and
  10 kJ mol\(^{-1}\) nm\(^{-1}\), respectively;
- restrained equilibration for 9 ps with v-rescale/Berendsen coupling, followed
  by 12.5 ps with Nose-Hoover/semi-isotropic Parrinello-Rahman coupling;
- 200 ns production at 303 K and 1 bar using a 20 fs timestep, v-rescale
  thermostat, and isotropic Parrinello-Rahman barostat;
- Verlet neighbour lists every 20 steps, reaction-field electrostatics, and
  shifted 1.1 nm Coulomb and van der Waals cutoffs;
- compressed coordinates and log/energy records every 1 ns, uncompressed coordinates,
  velocities, and forces every 10 ns, and energy calculation every 2 ps.

The production template states `constraints=none`, but this must not be
interpreted as an unconstrained system. Explicit Martini peptide and
polarizable-water topology constraints remained active and were handled by
LINCS (realized production order 4, one iteration). Position restraints with
force constant 4000 kJ mol\(^{-1}\) nm\(^{-2}\) were active during solvated
minimisation and equilibration and absent from production.

Peptide termini were charged by `martinize.py`; no explicit pH override was
used. Every residue was assigned extended secondary structure (`E`) at topology
generation, producing extended-state backbone parameters and local elastic
bonds. Completed production logs for 150 locally retained simulations all
reported GROMACS 2023.2 and the same realized production timing and output
intervals.

Legacy AP-SASA at time \(t\) is

\[
\mathrm{AP}_{SASA}^{legacy}(t)=\frac{SASA(0\ \mathrm{ns\ production})}{SASA(t)}.
\]

GROMACS `sasa` is applied to a named non-solvent peptide group. The main legacy value used for operational labels is the 200 ns ratio.

The intended paper-style AP-SASA is

\[
\mathrm{AP}_{SASA}^{paper}=
\frac{SASA_{\mathrm{initial,noncontact}}}
{\overline{SASA}_{190-200\mathrm{ns}}}.
\]

The initial source is preferably the inserted non-contact structure, with pre-production or production-start fallbacks explicitly recorded.

**Implementation caveat.** The current `paper_ap_sasa_last10ns_mean` field is calculated as the mean of framewise ratios,
\(\overline{SASA_0/SASA_t}\), while the method text states
\(SASA_0/\overline{SASA_t}\). These are not exactly identical. This field therefore needs confirmation/correction before being called an exact paper-style reproduction.

For path-based AP-contact, the minimum inter-bead distance \(d_{ij}\) between peptide copies defines

\[
w(d)=
\begin{cases}
1,&d\le0.4\ \mathrm{nm},\\
\exp[-(10d-4)],&0.4<d<1.2\ \mathrm{nm},\\
0,&d\ge1.2\ \mathrm{nm}.
\end{cases}
\]

A Hamiltonian path through all peptide copies is scored by mean consecutive-edge weight, and the maximum path score is retained. Exact dynamic programming is used only for at most 16 copies; larger systems use deterministic beam search with width 256 and branching factor 8. The final metric is the mean over 11 frames at 190, 191, …, 200 ns.

The actual Phase 3 operational label rule was:

\[
y_{\mathrm{CGMD}}=
\mathbf 1\left[
\mathrm{AP}_{SASA}^{legacy}(200\mathrm{ns})\ge1.75
\ \land\
\overline{\mathrm{AP}_{contact,path}}^{190-200\mathrm{ns}}\ge0.5
\right].
\]

Human review recorded label, rubric, confidence, evidence summary/tags, notes, reviewer, time, and campaign provenance. Visual inspection and cluster/contact summaries were supporting evidence, but metrics did not automatically write labels. These labels represent the specified simulation and decision rule; they are not interchangeable with wet-lab truth.

## 13. Evaluation metrics

\[
\mathrm{Accuracy}=\frac{TP+TN}{N},\quad
\mathrm{Precision}=\frac{TP}{TP+FP},\quad
\mathrm{Recall}=\frac{TP}{TP+FN},
\]

\[
F1=\frac{2PR}{P+R},\qquad
\mathrm{BalancedAccuracy}=\frac{\mathrm{Sensitivity}+\mathrm{Specificity}}2.
\]

ROC-AUC is computed by positive-negative ranking with half credit for ties. PR-AUC is trapezoidal area under the precision-recall sequence sorted by decreasing score. Brier score is \(N^{-1}\sum_i(p_i-y_i)^2\). Log loss is mean binary cross-entropy. ECE-10 is the weighted mean absolute calibration gap over ten equal-width probability bins; MCE-10 is the maximum bin gap.

Selected-peptide diversity is mean pairwise normalized Levenshtein distance:

\[
D(x_i,x_j)=\frac{\mathrm{edit}(x_i,x_j)}
{\max(|x_i|,|x_j|)}.
\]

Jaccard overlap between selected sets \(A,B\) is
\[
J(A,B)=\frac{|A\cap B|}{|A\cup B|}.
\]

Positive yield is the fraction of selected peptides whose revealed label is 1. Phase 2 summaries report mean and standard deviation across five outer folds. No formal null-hypothesis significance test is currently implemented; comparisons use paired fold summaries, deltas versus random, ranges, standard deviations, and worst-fold evidence.

## 14. Recommended methodology figures and tables

### Figures

1. **Overall adaptive discovery workflow** — double column. Nodes: published dataset → nested-CV reproduction → replay benchmark → three isolated Phase 3 branches → GA proposals → CG-MD → human review → branch-local ingestion → terminal retraining/evaluation. Caption: “Scientific workflow separating retrospective strategy evaluation from prospective CG-MD feedback.”
2. **Feature representation workflow** — double column. Sequence branches to AP, SP, TSNE-SP, padding/masking, then model-family mapping. Caption: “Conversion of peptide sequences into aggregation-propensity and physicochemical tensors.”
3. **AP_SP architecture** — single/double column. Three AP recurrent branches plus Conv1D–BiLSTM SP branch → concatenation → sigmoid. Caption: “Hybrid AP_SP classifier used as the adaptive surrogate.”
4. **Retrospective replay protocol** — double column. Outer holdout, validation, replay seed, hidden pool; retrain → score → reveal five labels → repeat. Caption: “Labels are hidden rather than simulated; no peptides are generated.”
5. **Branch-isolated real AL** — double column. Three parallel ledgers with a shared provenance-only inventory and no cross-branch arrows. Caption: “Prospective branch-specific feedback prevents label leakage between strategies.”
6. **GA proposal** — double column. Random population → strategy utility/penalties → tournament → crossover/mutation → parent+offspring survival → deduplication → acquisition selection.
7. **CG-MD evidence/review** — double column. PDB → Martini/PW → 200 ns → PBC preprocessing → SASA/contact/cluster/VMD evidence → human operational label → ingest.
8. **Self-paced workflow and result** — show the familiarity percentile, static and scheduled eligibility, the four trajectories, and the completed AULC comparison.

### Tables

1. **Dataset and split summary** — use verified values from Section 2.
2. **Model-family representations and frozen hyperparameters** — representation, branches, recurrent cells, kernel.
3. **Training and calibration settings** — optimizer, LR schedule, batch, epochs, class weights, calibration, threshold.
4. **Acquisition-function taxonomy** — formula, evidence type, final selection, hyperparameters.
5. **Phase 2 strategy results** — initial labels, holdout AULC-F1, terminal F1, PR-AUC, Brier, labels to targets, diversity, Jaccard.
6. **Phase 3 round summary** — branch, round, selected sequences, positive labels, cumulative labelled count, validation metrics.
7. **Final Phase 3 performance** — branch, 275-label validation metrics and final frozen-holdout metrics from the completed archive.
8. **CG-MD protocol and operational thresholds** — force field, solvent, copies, box, duration, temperature, pressure, metric windows, label rule.

Existing principal figures:

- `thesis_results/02_replay/evidence/figures/benchmark_holdout_f1_initial_10_vs_labeled_peptides.svg`
- `thesis_results/02_replay/evidence/figures/benchmark_holdout_f1_initial_40_vs_labeled_peptides.svg`
- `thesis_results/02_replay/evidence/figures/benchmark_holdout_mean_AULC_F1_initial_40.svg`
- `thesis_results/02_replay/evidence/figures/benchmark_holdout_labels_to_f1_086_combined_initial_10_40.svg`
- `thesis_results/02_replay/evidence/figures/benchmark_pairwise_jaccard_heatmap_initial_10.svg`
- `thesis_results/02_replay/evidence/figures/benchmark_pairwise_jaccard_heatmap_initial_40.svg`
- `thesis_results/03_real_al_strategy_selection/strategy_performance_vs_diversity_clean.svg`
- `thesis_results/03_real_al_strategy_selection/strategy_jaccard_heatmap_presentation.svg`

## 15. Appendix / code reproducibility details

### Important code

| Purpose | File / function |
|---|---|
| Dataset and splits | `active_learning_thesis/dataset.py` |
| Phase 1 nested CV | `active_learning_thesis/phase1_reproduction.py` |
| Feature preparation/model training | `active_learning_thesis/predictive.py` |
| Original model definitions | `SA_ML_predictive/code/models.py` |
| Original training settings | `SA_ML_predictive/code/automate_training.py` |
| Metrics/calibration evaluation | `active_learning_thesis/metrics.py` |
| Acquisition rules | `active_learning_thesis/acquisition.py` |
| GA integration | `active_learning_thesis/generative.py` |
| GA operators | `SA_ML_generative/genetic_algorithm_library.py` |
| Phase 2 replay | `active_learning_thesis/phase2_replay.py` |
| Phase 3 orchestration | `active_learning_thesis/phase3_real_al.py` |
| Real-AL training/final evaluation | `active_learning_thesis/workflow.py` |
| CG-MD packaging/parsing | `active_learning_thesis/md_workflow.py` |
| BO-style utilities | `active_learning_thesis/discovery.py` |
| Optional complexity sidecar | `active_learning_thesis/optional_evaluator_study.py` |
| Phase 5 self-paced replay | `active_learning_thesis/phase5_self_paced.py` |

### Important outputs

- Phase 1: `thesis_results/01_reproduction/`
- Phase 2: `thesis_results/02_replay/`
- Phase 2 figures: `thesis_results/02_replay/evidence/figures/`
- Strategy selection: `thesis_results/03_real_al_strategy_selection/`
- Completed Phase 3 branches: `thesis_results/03_real_al/phase3_results_20260620/thesis_results/03_real_al/branches/<strategy>/`
- Completed Phase 3 comparisons: `thesis_results/03_real_al/phase3_results_20260620/thesis_results/03_real_al/comparison/`
- Final validation trajectory: `comparison/all_rounds_branch_metrics.csv`
- Final holdout comparison: `comparison/final_branch_holdout_metrics.csv`
- Final branch holdout: `branches/<strategy>/metrics/final_holdout.json`

### Principal commands

```bash
python -m active_learning_thesis phase1-reproduce --output-root thesis_results/01_reproduction

python -m active_learning_thesis phase2-replay \
  --phase1-root thesis_results/01_reproduction \
  --output-root thesis_results/02_replay \
  --outer-folds 1 2 3 4 5 --inner-fold 1 \
  --replay-seed-sizes 10 40 --batch-size 5 --max-rounds 20

python -m active_learning_thesis phase2-export \
  --input-root thesis_results/02_replay \
  --output-root thesis_results/02_replay/evidence

python -m active_learning_thesis phase3-real-al status \
  --output-root thesis_results/03_real_al

python -m active_learning_thesis phase3-real-al compare \
  --output-root thesis_results/03_real_al --round 8

python -m active_learning_thesis phase3-real-al finalize \
  --output-root thesis_results/03_real_al \
  --branch <strategy> --round 8 --evaluate-holdout

python -m active_learning_thesis run-discovery \
  --run-dir <branch-or-run-directory>

python -m active_learning_thesis phase5-self-paced init \
  --phase1-root thesis_results/01_reproduction \
  --phase2-root thesis_results/02_replay \
  --output-root thesis_results/05_self_paced_active_learning \
  --pbs-repo-root "$PWD"
```

The `run-discovery` example above is legacy/generic discovery support. It is
not the thesis Phase 4 fixed-surrogate workflow and must not be used to produce
Phase 4 thesis evidence.

### Environment and hardware

The pinned environment specifies Python 3.10.13, TensorFlow 2.10.1, NumPy 1.26.3, scikit-learn 1.3.0, SciPy 1.11.4, pandas 2.1.4, matplotlib 3.8.0, `seqprops` 1.0.3, CUDA toolkit 11.2, and cuDNN 8.1. The local inspection shell used Python 3.11.3, but scientific runs should be attributed to the pinned Conda environment.

Phase 3 proposal/finalization jobs were configured on SUPEK for one GPU, four CPUs, 40 GB memory, and five-hour walltime. CG-MD ran as Slurm jobs on BURA with GROMACS 2023.2 and Intel MPI 2021.17.1. Exact GPU model, CPU model, and node allocation for every run **need confirmation from scheduler logs**.

### Final reporting status

- Phase 1: executed and aggregated.
- Phase 2: executed and aggregated with figures and evidence tables.
- Phase 3: executed and aggregated. The completed archive contains 120 reviewed and ingested CG-MD rows and terminal validation/frozen-holdout results.
- Primary Phase 4: executed and aggregated. All six policies completed with five selected peptides each; 29 valid proposals have reconciled CG-MD evidence and the known length-25 MES proposal is excluded from yield denominators.
- Phase 4-D: executed and aggregated as a separate exploratory replicate.
- Phase 5: executed and aggregated. All 12 replay jobs reached 235 labels and
  the corrected aggregate contains full/partial AULCs, labels-to-target,
  terminal convergence, proxy-validity, diversity, yield, runtime, and figures.

## 16. Phase 5 SPAL-inspired self-paced replay

Phase 5 is a retrospective single-model neural adaptation of the easy-to-hard
acquisition principle. It is not an exact reproduction of Tang and Huang's
kernel SPAL optimiser, which jointly uses a kernel least-squares model, MMD
representativeness, alternating optimisation, ADMM, and quadratic programming.

The experiment compares random selection, ordinary predictive entropy, a
static easy-candidate entropy restriction, and a progressively relaxed
self-paced entropy policy. The official Phase 2 split, calibration, replay,
and evaluation protocol is reused. The reduced primary experiment uses outer
folds 1--3 as fold-level repetitions with overlapping training partitions;
they are not statistically independent replicates. It uses \(n_0=10\), batch
size five, and 45 acquisition steps, reaching all 235 replay-training rows and
producing 46 replay points per trajectory.

At each replay point, exactly one calibrated `AP_SP` model provides a
384-dimensional penultimate concatenation representation. The representation
is L2-normalised independently for every peptide. Candidate distance is
measured to the nearest currently revealed replay-training peptide in that
model's latent space. This is termed a calibrated neural-model
labelled-manifold familiarity proxy. Optional software support for more than
one member is not part of the reported primary experiment.

The resulting distance is an operational model-familiarity or
labelled-manifold-distance proxy. It is not intrinsic peptide complexity,
biological easiness, experimental difficulty, simulation cost, or a chemical
complexity score. Its validity is assessed post hoc by relating raw distance
to pre-query prediction loss and error after oracle labels are revealed.

The self-paced eligible fraction follows

\[
\lambda_t=\operatorname{clip}\left(0.30+0.70t/44,0.30,1\right),
\qquad t=0,\ldots,44.
\]

The initial evaluation is replay point \(L_0\), and its first acquisition uses
step \(t=0\). Random and ordinary predictive entropy export familiarity
diagnostics but never consume them during selection. Hidden labels are joined
only after selection for retrospective proxy validation.

Before every fit, currently revealed training rows are sorted by immutable
original dataset row identifier. Model seeds depend on outer fold, initial
count, replay point, and member index, but not strategy. At 235 labels all
strategies therefore share the same sequences, canonical row order, single
model seed, architecture, validation set, and training/calibration procedure.
TensorFlow deterministic operation mode is enabled where supported. Terminal
parameter, holdout-prediction, F1, and PR-AUC differences are audited against a
predefined \(10^{-6}\) numerical tolerance rather than requiring bitwise
identity across hardware.

The principal normalised discrete trapezoidal AULC-F1 covers 10--235 labels.
Preregistered descriptive partial AULCs cover 10--60, 10--110, and 10--160.
Positive-label yield, selected-sequence diversity, and pairwise strategy
overlap are reported per acquisition batch and cumulatively at every matched
labelled count, with fixed-budget summaries at 60, 110, and 160 labels.
Terminal cumulative yield, diversity, and overlap are consistency checks
because every strategy has then revealed the same 235 rows.

The fold-matched Phase 1 `AP_SP` F1 and PR-AUC are reported only as contextual
baselines. Those models use 294 non-holdout development rows, whereas Phase 5
fits at most 235 replay-training rows and reserves 59 rows for checkpoint
selection and member-wise calibration. Absolute metric gaps and the protocol
difference are reported; Phase 1 does not enter Phase 5 strategy rankings,
AULC calculations, or hypothesis tests.

### Completed Phase 5 results

Mean normalised AULC-F1 across the three overlapping fold conditions was:

| Strategy | 10--60 | 10--110 | 10--160 | 10--235 |
|---|---:|---:|---:|---:|
| predictive entropy | 0.7773 | 0.8049 | 0.8183 | **0.8223** |
| random | **0.7778** | 0.7982 | 0.8101 | 0.8171 |
| self-paced entropy | 0.7687 | 0.7944 | 0.8077 | 0.8147 |
| static easy entropy | 0.7723 | 0.7846 | 0.7915 | 0.8016 |

On the full interval, self-paced entropy exceeded static easy entropy by
0.01309 AULC-F1, but trailed predictive entropy by 0.00768 and random by
0.00242. Self-paced entropy reached F1 0.86 in all three folds at a conditional
mean of 71.7 labels. Predictive entropy reached the same target in all folds at
105 labels; random and static easy entropy reached it in two of three folds.
Because target crossings may be non-monotonic, the full AULC remains the
principal trajectory summary.

At 235 labels all four strategies within each fold had identical parameters,
holdout predictions, F1, and PR-AUC. Mean terminal holdout F1 was 0.7951 and
mean PR-AUC was 0.8995. The contextual fold-matched Phase 1 means were 0.8624
and 0.9276, respectively, under the different 294-row development protocol.

The familiarity proxy showed heterogeneous, trajectory-dependent association
with pre-query log loss. Mean fold-step Spearman correlation was 0.3865 for
static easy entropy, 0.2399 for random, 0.2095 for self-paced entropy, and
0.0099 for predictive entropy, with wide ranges that included negative values.
It should therefore remain described as an operational familiarity proxy.

The thesis-safe conclusion is that progressive pacing was preferable to a
fixed easy-only restriction, but did not improve overall retrospective label
efficiency over ordinary predictive entropy or random selection in this
reduced three-fold experiment.
