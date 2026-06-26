# Machine learning for peptide self-assembly
[![DOI](https://zenodo.org/badge/861589296.svg)](https://doi.org/10.5281/zenodo.13847868)
> <div align="justify">Supramolecular peptide-based materials have great potential for revolutionizing fields like nanotechnology and medicine. However, deciphering the intricate sequence-to-assembly pathway, essential for their real-life applications, remains a challenging endeavour. Their discovery relies primarily on empirical approaches that require substantial financial resources, impeding their disruptive potential. Consequently, despite the multitude of characterized self-assembling peptides and their demonstrated advantages, only a few peptide materials have found their way to the market. Machine learning trained on experimentally verified data presents a promising tool for quickly identifying sequences with a high propensity to self-assemble, thereby focusing resource expenditures on the most promising candidates. Here we introduce a framework that implements an accurate classifier in a metaheuristic-based generative model to navigate the search through the peptide sequence space of challenging size. For this purpose, we trained five recurrent neural networks among which the hybrid model that uses sequential information on aggregation propensity and specific physicochemical properties achieved a superior performance with 81.9% accuracy and 0.865 F1 score. Molecular dynamics simulations and experimental validation have confirmed the generative model to be 80–95% accurate in the discovery of self-assembling peptides, outperforming the current state-of-the-art models. The proposed modular framework efficiently complements human intuition in the exploration of self-assembling peptides and presents an important step in the development of intelligent laboratories for accelerated material discovery.</div>

## System requirements
A standard desktop computer with 4+ GB RAM and a single-core, 2+ GHz CPU should suffice to run the algorithms. No non-standard hardware is necessary. `Conda` package manager is required to set up a Python environment and install dependencies. Software dependencies along with version numbers are detailed in the ```ml_peptide_self_assembly.yml``` file. The code was tested on Ubuntu 22.04 and Windows 11 operating systems.

## Environment setup &amp; activation
Use `conda` and `ml_peptide_self_assembly.yml` to create an environment. This usually takes about 5 minutes.

    git clone https://github.com/mnjirjak/ml_peptide_self_assembly.git
    cd ml_peptide_self_assembly
    conda env create -f ml_peptide_self_assembly.yml
    conda activate ml_peptide_self_assembly

## Important notes
- The code should be run from the root (```ml_peptide_self_assembly```) directory with the conda environment active.
- If you want to train your model and perform hyperparameter optimization, modify the peptide list and labels in ```SA_ML_predictive/data/data_SA.csv```, and alter the hyperparameter values in the header of ```SA_ML_predictive/code/automate_training.py``` before running the scripts.
- Prediction time varies with the model used but usually takes about 15 seconds.
- The generative model takes about 5 minutes to suggest novel peptides.

## Practical runbooks

For the thesis workflow and cluster usage, see:

- [`THESIS_HANDOFF_FOR_NEXT_MODEL.md`](THESIS_HANDOFF_FOR_NEXT_MODEL.md) - authoritative current thesis context, phase status, and artefact map
- [`ACTIVE_LEARNING_THESIS.md`](ACTIVE_LEARNING_THESIS.md)
- [`SUPEK_RUNBOOK.md`](SUPEK_RUNBOOK.md)
- [`BURA_MD_RUNBOOK.md`](BURA_MD_RUNBOOK.md)

Verified thesis status on 22 June 2026: Phases 1, 2, 3, primary Phase 4,
Phase 4-D, and Phase 5 are executed and aggregated. Phase 5 results are
summarised in [`PHASE5_RESULTS_SUMMARY.md`](PHASE5_RESULTS_SUMMARY.md). Use the
handoff above rather than inferring status from older preview directories.

## Admin dashboard

A local Streamlit operator dashboard is available for monitoring active-learning runs, latest proposed peptides, MD ladder progress, and the next recommended commands:

    python -m active_learning_thesis dashboard --run-root active_learning_runs

Use `--host`, `--port`, and `--refresh-seconds` if you want to expose it on a different interface or auto-refresh the page while you monitor a long run.

The dashboard is now a local admin cockpit in Phase 3:
- it can launch safe local workflow commands such as `prepare-md-stage`,
  `finalize-md-stage`, `make-md-ingest-csv`, `propose-round`, the legacy/generic
  `run-discovery` helper, and `evaluate-final`
- it can draft, approve, and track allowlisted remote actions for Supek and BURA using system `ssh`/`scp`
- mutating remote actions remain human-approved; polling actions stay read-only
- it records local and remote action history under `active_learning_runs/_dashboard_actions/` with stdout/stderr logs
- it stores remote sync state and queue snapshots under `active_learning_runs/_dashboard_remote_state/`
- cluster connection profiles live outside the repo at `%USERPROFILE%\\.active_learning_thesis\\cluster_profiles.json` by default

The generic `run-discovery` command is not the thesis Phase 4 experiment.
Phase 4 uses the dedicated `phase4-bo` workflow and its fixed-surrogate audit
under `thesis_results/04_bayesian_optimization/`.
## Predictive model(s)
Predictive models take a peptide sequence, represented by amino acid single-letter codes, and output the probability of the sequence exhibiting self-assembly.

                   ┌────────┐
    KFFAKK ───────►│ML model├─────► 98.3%
                   └────────┘

#### Use one of the existing, pre-trained models:

    python SA_ML_predictive/code/predict.py --sequence <peptide_sequence> --ml-model <AP, AP_SP, SP, TSNE_AP_SP, TSNE_SP>

The resulting probability of a single sequence exhibiting self-assembly will be visible in the command line.

#### Running hyperparameter selection for the model(s):

    python SA_ML_predictive/code/hyperparameter_selection.py --ml-model <AP, AP_SP, SP, TSNE_AP_SP, TSNE_SP>

#### Train the model(s) with the chosen hyperparameters:

    python SA_ML_predictive/code/training.py --ml-model <AP, AP_SP, SP, TSNE_AP_SP, TSNE_SP> --num-cells <an integer value> --kernel-size <an integer value>

The resulting model will be saved in the ```SA_ML_predictive/models``` directory and will overwrite an existing model.

## Generative model
The generative model is a machine-learning-guided genetic algorithm capable of efficiently exploring peptide chemical space and creating novel sequences.

    ┌────────┐
    │GA model├─────► FKFEFFKF, GFALLAGKK, AAVVLWEEEE, ...
    └────────┘

#### Use the generative model with one of the existing predictive models:

    python SA_ML_generative/find_novel_peptides.py --ml-model <AP, AP_SP, SP, TSNE_AP_SP, TSNE_SP>

#### Use the generative model with your predictive model:

    python SA_ML_generative/find_novel_peptides.py --ml-model-path <path to your .h5 file>

The results will be saved in a ```csv``` file. Additional parameters of the generative model, such as mutation probability or a list of allowed amino acids, can be modified in the header of the ```SA_ML_generative/find_novel_peptides.py``` script.

## Citation
If you use the software or data published in this repository, please cite the following article:
```
@article{Njirjak_Reshaping_the_discovery_2024,
  author = {Njirjak, Marko and Žužić, Lucija and Babić, Marko and Janković, Patrizia and Otović, Erik and Kalafatovic, Daniela and Mauša, Goran},
  doi = {10.1038/s42256-024-00928-1},
  journal = {Nature Machine Intelligence},
  title = {{Reshaping the discovery of self-assembling peptides with generative AI guided by hybrid deep learning}},
  year = {2024}
}
```





