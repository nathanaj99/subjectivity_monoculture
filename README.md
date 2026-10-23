# The Subjectivity of Monoculture: Replication Files

Replication code and data for

> Nathanael Jo, Nikhil Garg, and Manish Raghavan. **The Subjectivity of Monoculture.** In *Advances in Neural Information Processing Systems (NeurIPS)*, 2026. https://arxiv.org/abs/2602.24086

```bibtex
@inproceedings{jo2026subjectivity,
  title     = {The Subjectivity of Monoculture},
  author    = {Jo, Nathanael and Garg, Nikhil and Raghavan, Manish},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## 1. Quick start (reproduce the figures)

The fitted model parameters are already in `results/`, so **reproducing every figure only requires the scripts in `scripts/analysis/`**. You do not need to rerun `scripts/experiments/`, which takes a few hours on a CPU.

```bash
git lfs install && git lfs pull        # data/ is stored with Git LFS
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run all scripts from the repository root (paths are relative to it)
for f in fig2 fig3 fig4_7 fig5 fig6; do python scripts/analysis/$f.py; done
```

Figures are written to `figs/`. On an Apple M2 Pro laptop (CPU only) the analysis scripts take about 4 minutes in total: `fig2.py` and `fig6.py` take about 1.5 minutes each, mostly to compute the pairwise agreement metrics, and the others take under 20 seconds each.

## 2. Requirements

- Python 3.11. Other 3.10+ versions should work, but only 3.11 was tested.
- The packages in `requirements.txt`: numpy, pandas, scipy, matplotlib, seaborn, scikit-learn, tqdm, and torch.
  - **numpy ≥ 2** is required, because the pickles in `results/` were written with NumPy 2.
  - **pandas < 3** is required, because pandas 3's copy-on-write makes `DataFrame.values` read-only, which breaks `np.fill_diagonal` in `fig2.py`/`fig6.py`.
  - torch is only needed for `scripts/experiments/`.
- [Git LFS](https://git-lfs.com/) to download `data/` (about 430 MB).
- No GPU is needed. The experiment scripts use CUDA automatically when it is available.

## 3. Repository structure

```
.
├── data/                      # Inputs (Git LFS)
│   ├── acsincome/             # ACSIncome features/labels, California 2018 1-year ACS
│   ├── helm/                  # HELM MMLU per-model, per-question responses
│   └── hf/                    # Hugging Face Open LLM Leaderboard per-model, per-question responses + model metadata
├── results/                   # Outputs of scripts/experiments (fitted parameters, model predictions)
│   ├── helm/
│   ├── hf/
│   └── relativity_population/ # Experiments for Fig. 3 (ACSIncome + HELM subsets)
├── scripts/
│   ├── experiments/           # Model fitting (slow)
│   └── analysis/              # Figure generation (fast)
└── figs/                      # Generated figures
```

### Data

| File | Description |
|---|---|
| `data/helm/helm_ans.csv` | Binary correctness matrix: 14,042 MMLU questions × 71 models (`question_id` + one column per model). |
| `data/helm/helm_mult_choice_ans.csv` | One-hot version of the HELM responses (one row per question × answer option). |
| `data/helm/helm_all_mmlu_data_limitedcols.csv` | Long format: model, subject, question id, correct letter, predicted letter, correctness. Used for the Kim et al. and Goel et al. agreement metrics. |
| `data/helm/helm_mmlu.csv` | HELM correctness matrix with a larger set of models (not used by the current scripts). |
| `data/hf/hf_ans.csv` | Binary correctness matrix: 11,994 questions × 451 Open LLM Leaderboard models. |
| `data/hf/hf_ans_with_metadata.csv` | `hf_ans.csv` plus question text/options (`doc`), `doc_hash`, `src`, and `category`. |
| `data/hf/hf_mult_choice_ans.csv` | One-hot version of the HF responses (one row per question × answer option). |
| `data/hf/hf_model_answers.txt` | Each model's predicted option per question, plus the correct answer (used for agreement metrics). |
| `data/hf/hugging_face.csv`, `hf_normalized.csv`, `hf_covariates_scaled.csv` | Leaderboard model metadata/covariates (not used by the current scripts). |
| `data/acsincome/CA_2018_1yr.csv` | ACSIncome task (10 features, `label`, `group`) for California, 2018 1-year ACS, as produced by [folktables](https://github.com/socialfoundations/folktables). |

## 4. Scripts

### Analysis (`scripts/analysis/`)

Each `fig*.py` script generates the correspondingly numbered figure(s) in the paper (`fig4_7.py` generates Figs. 4 and 7) from `data/` and `results/`, and saves them to `figs/`.

### Experiments (`scripts/experiments/`)

Run these from the repository root, e.g. `python scripts/experiments/infer_q_d.py`.

- `infer_q_d.py`: fits the one-dimensional probit IRT model (model ability and item difficulty) on HELM and HF (Appendix C.1).
- `infer_q_d_multidimensional.py`: fits the multidimensional IRT (MIRT) model on HELM and HF for a range of dimensions K (Appendix C.1).
- `acsincome_models.py`: trains a collection of classifiers on ACSIncome and records which test examples each one gets right, plus a set of artificial independent models.
- `population_relativity.py`: fits the IRT model on nested populations of models from ACSIncome and HELM, for Fig. 3. Run it after `acsincome_models.py`.

## 5. Notes on reproducibility

- Optimization starts from random initializations and model training is not seeded everywhere, so rerunning `scripts/experiments/` gives numerically slightly different parameters (and slightly different figures). The qualitative conclusions are unchanged.
