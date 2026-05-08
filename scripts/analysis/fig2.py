import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys
from tqdm import tqdm
from scipy.stats import norm
import ast
from itertools import combinations
from typing import Tuple


def pairwise_kim_excess_and_capa(
    df: pd.DataFrame,
    k_options: int = 4,
    model_col: str = "model",
    qid_col: str = "question_id",
    pred_col: str = "predicted_letter",
    correct_bool_col: str = "got_correct_answer",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      kim_excess:  m x m symmetric DataFrame with
        P(pred1 == pred2 | both wrong) - 1/(K-1)

      capa:        m x m symmetric DataFrame with CAPA (Eq. 4),
        using discrete (one-hot) probabilities derived from predicted_letter.
    """
    required = {model_col, qid_col, pred_col, correct_bool_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only needed cols, drop duplicates if any
    d = df[[model_col, qid_col, pred_col, correct_bool_col]].drop_duplicates()

    models = sorted(d[model_col].unique().tolist())
    m = len(models)

    kim_excess = pd.DataFrame(np.nan, index=models, columns=models, dtype=float)
    capa = pd.DataFrame(np.nan, index=models, columns=models, dtype=float)

    # Fill diagonals: Kim excess is 1 - 1/(K-1) when conditioning on "both wrong"
    # but on-diagonal is a bit degenerate (a model always agrees with itself);
    # we set diagonal to NaN by default. For CAPA, set 1.0 on diagonal.
    np.fill_diagonal(capa.values, 1.0)

    baseline_wrong_agreement = 1.0 / (k_options - 1)

    # Pre-split by model for speed
    by_model = {
        mod: sub[[qid_col, pred_col, correct_bool_col]].copy()
        for mod, sub in d.groupby(model_col, sort=False)
    }

    for a, b in combinations(models, 2):
        da = by_model[a]
        db = by_model[b]

        merged = da.merge(db, on=qid_col, how="inner", suffixes=("_a", "_b"))
        if merged.empty:
            # leave as NaN if no shared questions
            continue

        pred_a = merged[f"{pred_col}_a"].to_numpy()
        pred_b = merged[f"{pred_col}_b"].to_numpy()
        corr_a = merged[f"{correct_bool_col}_a"].to_numpy(dtype=bool)
        corr_b = merged[f"{correct_bool_col}_b"].to_numpy(dtype=bool)

        # ---------- (1) Kim et al.: agreement conditional on both wrong ----------
        both_wrong = (~corr_a) & (~corr_b)
        if both_wrong.any():
            agree_when_wrong = np.mean(pred_a[both_wrong] == pred_b[both_wrong])
            kim_val = (agree_when_wrong - baseline_wrong_agreement) / (1 - baseline_wrong_agreement)
        else:
            kim_val = np.nan  # undefined if no co-errors

        kim_excess.loc[a, b] = kim_val
        kim_excess.loc[b, a] = kim_val

        # ---------- (2) Goel et al.: CAPA (Eq. 4) with Eq. (2),(3) ----------
        # Eq. (2): c_obs^p = (1/n) sum_i sum_k p1ik p2ik.
        # With one-hot probs from argmax, sum_k p1ik p2ik = 1{pred_a == pred_b}.
        c_obs = np.mean(pred_a == pred_b)

        # p^c_j: average probability assigned to the correct option.
        # With one-hot probs, that's exactly accuracy.
        p1c = np.mean(corr_a)
        p2c = np.mean(corr_b)

        # Eq. (3): independent baseline with p^c on correct option
        # and uniform over (K-1) incorrect options.
        # Expected agreement = p1c*p2c + (1-p1c)(1-p2c)/(K-1).
        c_exp = (p1c * p2c) + ((1 - p1c) * (1 - p2c)) / (k_options - 1)

        # Eq. (4): CAPA = (c_obs - c_exp) / (1 - c_exp)
        denom = 1.0 - c_exp
        capa_val = np.nan if np.isclose(denom, 0.0) else (c_obs - c_exp) / denom

        capa.loc[a, b] = capa_val
        capa.loc[b, a] = capa_val

    return kim_excess, capa



def pairwise_kim_excess_and_capa_variable_k(
    df_wide: pd.DataFrame,
    question_id_col: str = "question_id",
    correct_col: str = "correct_answer",
    k_col: str = "num_options",
    model_cols: list[str] | None = None,
    drop_invalid_k: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Wide df input:
      - columns include: question_id_col, correct_col, k_col
      - remaining columns (or model_cols if provided) are model predictions (argmax choices)

    Returns:
      kim_excess: m x m symmetric DataFrame with
        P(pred1 == pred2 | both wrong) - avg_i[1/(K_i-1) | both wrong]

      capa: m x m symmetric DataFrame with CAPA (Eq. 4),
        using one-hot/argmax reduction and variable K_i in the chance term.

    Notes:
      - Pairwise computations use the intersection of rows where both models have non-null predictions
        and correct_answer / num_options are present.
      - If a pair has no co-errors, kim_excess is NaN for that pair.
      - If 1 - c_exp is 0 (degenerate), CAPA is NaN for that pair.
    """
    required = {question_id_col, correct_col, k_col}
    missing = required - set(df_wide.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Decide model columns
    if model_cols is None:
        model_cols = [c for c in df_wide.columns if c not in [question_id_col, correct_col, k_col]]
    if len(model_cols) == 0:
        raise ValueError("No model prediction columns found (or model_cols is empty).")

    # Work on a copy with index for alignment
    d = df_wide[[question_id_col, correct_col, k_col] + model_cols].copy()
    d = d.drop_duplicates(subset=[question_id_col]).set_index(question_id_col, drop=True)

    # Ensure K is numeric
    d[k_col] = pd.to_numeric(d[k_col], errors="coerce")

    if drop_invalid_k:
        # Keep only rows with K >= 3 (since K-1 in denominators)
        d = d[(d[k_col].notna()) & (d[k_col] >= 3)]

    models = list(model_cols)
    m = len(models)

    kim_excess = pd.DataFrame(np.nan, index=models, columns=models, dtype=float)
    capa = pd.DataFrame(np.nan, index=models, columns=models, dtype=float)
    np.fill_diagonal(capa.values, 1.0)  # CAPA of a model with itself is 1 in this discrete setting

    # Pre-extract arrays/series for speed
    correct = d[correct_col]
    K = d[k_col]

    preds = {mod: d[mod] for mod in models}

    for a, b in combinations(models, 2):
        pa = preds[a]
        pb = preds[b]

        # Shared valid rows: both predictions present, correct present, K present/valid
        mask = pa.notna() & pb.notna() & correct.notna() & K.notna() & (K >= 3)
        if not mask.any():
            continue

        pa_m = pa[mask]
        pb_m = pb[mask]
        corr_m = correct[mask]
        K_m = K[mask].astype(float)

        # correctness indicators (argmax-based)
        ya = (pa_m.values == corr_m.values)
        yb = (pb_m.values == corr_m.values)

        # ---------------- (1) Kim excess: conditional on both wrong ----------------
        both_wrong = (~ya) & (~yb)
        if both_wrong.any():
            agree_when_wrong = np.mean(pa_m.values[both_wrong] == pb_m.values[both_wrong])
            baseline_when_wrong = np.mean(1.0 / (K_m.values[both_wrong] - 1.0))
            kim_val = (agree_when_wrong - baseline_when_wrong) / (1 - baseline_when_wrong)
        else:
            kim_val = np.nan

        kim_excess.loc[a, b] = kim_val
        kim_excess.loc[b, a] = kim_val

        # ---------------- (2) CAPA with variable K ----------------
        # observed agreement (Eq. 2 under one-hot reduction)
        c_obs = np.mean(pa_m.values == pb_m.values)

        # p^c_j estimated as accuracy on shared items for this pair (discrete proxy)
        p1c = float(np.mean(ya))
        p2c = float(np.mean(yb))

        # expected agreement: average over items of itemwise chance term
        # c_exp_i = p1c*p2c + (1-p1c)(1-p2c)/(K_i-1)
        inv_wrong = 1.0 / (K_m.values - 1.0)
        c_exp = float(np.mean((p1c * p2c) + ((1.0 - p1c) * (1.0 - p2c)) * inv_wrong))

        denom = 1.0 - c_exp
        capa_val = np.nan if np.isclose(denom, 0.0) else (c_obs - c_exp) / denom

        capa.loc[a, b] = capa_val
        capa.loc[b, a] = capa_val

    return kim_excess, capa



###### --- PROCESS HELM --- ######
helm = pd.read_csv('data/helm/helm_all_mmlu_data_limitedcols.csv')
helm['question_id'] = helm['subject'] + '-' + helm['id']
excess, capa = pairwise_kim_excess_and_capa(helm)

P = data.mean(axis=0)

# MSE on probabilities (same as before)
mse = np.mean((P - data) ** 2)

# --- Raw residuals (no Pearson scaling) ---
E = data - P                                # (n, m)

# Debias by removing column means (across items i)
E_centered = E - E.mean(axis=0, keepdims=True)

n = E.shape[0]

# Raw residual covariance (population-style, divide by n)
C = (E_centered.T @ E_centered) / n         # (m, m), set diag to 0 later if desired

# Residual variances across subjects (items x items diagonal)
var = np.mean(E_centered**2, axis=0)   # shape (m,)
std = np.sqrt(var)

# Outer product of stds
denom = np.outer(std, std)

# Avoid division by zero
with np.errstate(divide='ignore', invalid='ignore'):
    R = C / denom
    R[denom == 0] = 0.0


with open('results/helm_mult_choice_latent_p15.pkl', 'rb') as f:
    params = pickle.load(f)
mu = params['q'][None, :] - params['d'][:, None]

P = norm.cdf(mu)                            # fitted probs

# MSE on probabilities (same as before)
mse = np.mean((P - data) ** 2)

# --- Raw residuals (no Pearson scaling) ---
E = data - P                                # (n, m)

# Debias by removing column means (across items i)
E_centered = E - E.mean(axis=0, keepdims=True)

# n = E.shape[0]

# Raw residual covariance (population-style, divide by n)
C = (E_centered.T @ E_centered) / n         # (m, m), set diag to 0 later if desired

# Residual variances across subjects (items x items diagonal)
var = np.mean(E_centered**2, axis=0)   # shape (m,)
std = np.sqrt(var)

# Outer product of stds
denom = np.outer(std, std)

# Avoid division by zero
with np.errstate(divide='ignore', invalid='ignore'):
    R_1d = C / denom
    R_1d[denom == 0] = 0.0



###### --- PROCESS HF --- ######
hf = pd.read_csv('data/hf/hf_model_answers.txt', delimiter=',')
hf = hf.merge(df[['doc_hash', 'num_options']], on='doc_hash', how='left')
hf = hf.rename(columns={'doc_hash': 'question_id'})
excess_hf, capa_hf = pairwise_kim_excess_and_capa_variable_k(hf)



df = pd.read_csv('data/hf/hf_ans_with_metadata.csv')
df_subset = df[df.columns[1:-4]].to_numpy()
# df['doc'] = df['doc'].apply(ast.literal_eval)
# df['num_options'] = df['doc'].apply(lambda x: len(x['options']))
# hf = hf.merge(df[['doc_hash', 'num_options']], on='doc_hash', how='left')
# hf = hf.rename(columns={'doc_hash': 'question_id'})

# df_subset = df.drop(columns=['question_id']).to_numpy()

P = df_subset.mean(axis=0)

# MSE on probabilities (same as before)
mse = np.mean((P - df_subset) ** 2)

# --- Raw residuals (no Pearson scaling) ---
E = df_subset - P                                # (n, m)

# Debias by removing column means (across items i)
E_centered = E - E.mean(axis=0, keepdims=True)

# n = E.shape[0]

# Raw residual covariance (population-style, divide by n)
C = (E_centered.T @ E_centered) / n         # (m, m), set diag to 0 later if desired

# Residual variances across subjects (items x items diagonal)
var = np.mean(E_centered**2, axis=0)   # shape (m,)
std = np.sqrt(var)

# Outer product of stds
denom = np.outer(std, std)

# Avoid division by zero
with np.errstate(divide='ignore', invalid='ignore'):
    R_hf = C / denom
    R_hf[denom == 0] = 0.0

df = pd.read_csv('data/hf/hf_mult_choice_ans.csv')
df_subset = df.drop(columns=['question_id']).to_numpy()
with open(f'results/hf_mult_choice.pkl', 'rb') as f:
    params = pickle.load(f)
q_hat = params['q']
d_hat = params['d']

m = q_hat.shape[0]
n = d_hat.shape[0]

mu = q_hat[None, :] - d_hat[:, None]   # (n, m)
P = norm.cdf(mu)                            # fitted probs

# MSE on probabilities (same as before)
mse = np.mean((P - df_subset) ** 2)

# --- Raw residuals (no Pearson scaling) ---
E = df_subset - P                                # (n, m)

# Debias by removing column means (across items i)
E_centered = E - E.mean(axis=0, keepdims=True)

# n = E.shape[0]

# Raw residual covariance (population-style, divide by n)
C = (E_centered.T @ E_centered) / n         # (m, m), set diag to 0 later if desired

# Residual variances across subjects (items x items diagonal)
var = np.mean(E_centered**2, axis=0)   # shape (m,)
std = np.sqrt(var)

# Outer product of stds
denom = np.outer(std, std)

# Avoid division by zero
with np.errstate(divide='ignore', invalid='ignore'):
    R_1d_hf = C / denom
    R_1d_hf[denom == 0] = 0.0


df_subset = df[df.columns[1:-5]].to_numpy()

P = df_subset.mean(axis=0)

# MSE on probabilities (same as before)
mse = np.mean((P - df_subset) ** 2)

# --- Raw residuals (no Pearson scaling) ---
E = df_subset - P                                # (n, m)

# Debias by removing column means (across items i)
E_centered = E - E.mean(axis=0, keepdims=True)

# n = E.shape[0]

# Raw residual covariance (population-style, divide by n)
C = (E_centered.T @ E_centered) / n         # (m, m), set diag to 0 later if desired

# Residual variances across subjects (items x items diagonal)
var = np.mean(E_centered**2, axis=0)   # shape (m,)
std = np.sqrt(var)

# Outer product of stds
denom = np.outer(std, std)

# Avoid division by zero
with np.errstate(divide='ignore', invalid='ignore'):
    R_hf = C / denom
    R_hf[denom == 0] = 0.0


k = 1
with open(f'results/hf/hf_k{k}_p15.pkl', 'rb') as f:
    params = pickle.load(f)
q_hat = params['q']
d_hat = params['d']

m = q_hat.shape[0]
n = d_hat.shape[0]

mu = q_hat[None, :] - d_hat[:, None]   # (n, m)
P = norm.cdf(mu)                            # fitted probs

# MSE on probabilities (same as before)
mse = np.mean((P - df_subset) ** 2)

# --- Raw residuals (no Pearson scaling) ---
E = df_subset - P                                # (n, m)

# Debias by removing column means (across items i)
E_centered = E - E.mean(axis=0, keepdims=True)

# n = E.shape[0]

# Raw residual covariance (population-style, divide by n)
C = (E_centered.T @ E_centered) / n         # (m, m), set diag to 0 later if desired

# Residual variances across subjects (items x items diagonal)
var = np.mean(E_centered**2, axis=0)   # shape (m,)
std = np.sqrt(var)

# Outer product of stds
denom = np.outer(std, std)

# Avoid division by zero
with np.errstate(divide='ignore', invalid='ignore'):
    R_1d_hf = C / denom
    R_1d_hf[denom == 0] = 0.0


import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# ----- your settings -----
vmin, vmax = -0.2, 0.6
cmap_name = "RdBu_r"

# HELM (first 3)
data_list_helm = [excess, capa, R, R_1d]
titles_helm = ["Kim et al.", "Goel et al.", r"IRT-0.5", "IRT-1"]

# HF (last 3)
data_list_hf = [
    excess_hf.to_numpy()[:100, :100],
    capa_hf.to_numpy()[:100, :100],
    R_hf[:100, :100],
    R_1d_hf[:100, :100],
]
titles_hf = ["Kim et al.", "Goel et al.", r"IRT-0.5", "IRT-1"]

data_list = data_list_helm + data_list_hf
titles = titles_helm + titles_hf

# ----- layout: 6 plots + colorbar, with a spacer column between 3 and 4 -----
fig = plt.figure(figsize=(30, 4), constrained_layout=False)

# columns: [0][1][2][spacer][3][4][5][colorbar]
# make spacer a bit wide; make cbar slim
gs = GridSpec(
    1, 10, figure=fig,
    width_ratios=[1, 1, 1, 1, 0.005, 1, 1, 1, 1, 0.08],
    wspace=0.03
)

axes = []
axes.append(fig.add_subplot(gs[0, 0]))
axes.append(fig.add_subplot(gs[0, 1]))
axes.append(fig.add_subplot(gs[0, 2]))
axes.append(fig.add_subplot(gs[0, 3]))
# gs[0, 3] is the spacer (unused)
# axes.append(fig.add_subplot(gs[0, 4]))
axes.append(fig.add_subplot(gs[0, 5]))
axes.append(fig.add_subplot(gs[0, 6]))
axes.append(fig.add_subplot(gs[0, 7]))
axes.append(fig.add_subplot(gs[0, 8]))
cax = fig.add_subplot(gs[0, 9])  # colorbar axis

# ----- draw heatmaps (no individual colorbars) -----
for ax, data, title in zip(axes, data_list, titles):
    arr = np.asarray(data)

    # set diagonal to NaN
    np.fill_diagonal(arr, np.nan)

    sns.heatmap(
        arr,
        ax=ax,
        cmap=cmap_name,
        center=0,
        square=True,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        xticklabels=False,
        yticklabels=False
    )
    ax.set_title(title, fontsize=22)
    ax.set_xlabel("")
    ax.set_ylabel("")

# ----- one shared colorbar on the right -----
mappable = axes[0].collections[0]
cbar = fig.colorbar(mappable, cax=cax)

# make tick labels larger
cbar.ax.tick_params(labelsize=16)

cbar.ax.text(
    -2.5, 1.15,              # (x, y) in axes coords
    "Correlation",
    transform=cbar.ax.transAxes,
    ha="left",
    va="top",
    fontsize=19
)

# cbar.ax.set_title(
#     "Correlation",
#     fontsize=20,
#     pad=15
# )

# add colorbar label
# cbar.set_label(
#     "Correlation",
#     fontsize=16,
#     rotation=90,
#     labelpad=10
# )

# ----- dataset labels (bold, rotated) -----
# Put HELM just left of the first panel; HF just left of the 4th panel
# axes[0].text(
#     -0.05, 0.5, "HELM",
#     transform=axes[0].transAxes,
#     rotation=90,
#     va="center", ha="center",
#     fontweight="bold", fontsize=18
# )
# axes[4].text(
#     -0.05, 0.5, "HF",
#     transform=axes[4].transAxes,
#     rotation=90,
#     va="center", ha="center",
#     fontweight="bold", fontsize=18
# )

# -----------------------
# Panel labels (a) and (b)
# -----------------------
axes[0].text(
    0.02, 1.22, "(a) HELM",
    transform=axes[0].transAxes,
    ha="left", va="top",
    fontsize=22, fontweight="bold"
)
axes[4].text(
    0.02, 1.22, "(b) HF",
    transform=axes[4].transAxes,
    ha="left", va="top",
    fontsize=22, fontweight="bold"
)

# HELM: first three subplots
add_group_brace(
    fig,
    y_offset=-0.02,
    text_offset=-0.03,
    axes_group=axes[0:3],
    text="No item-level heterogeneity"
)

# HF: first three subplots in panel (b)
add_group_brace(
    fig,
    y_offset=-0.02,
    text_offset=-0.03,
    axes_group=axes[4:7],
    text="No item-level heterogeneity"
)



# ----- vertical delimiter between panel 3 and 4 -----
# Draw it in figure coordinates so it spans the full heatmap height region.
fig.canvas.draw()  # ensure positions are computed
bbox3 = axes[3].get_position()
bbox4 = axes[4].get_position()

x = (bbox3.x1 + bbox4.x0) / 2.0  # center of the spacer gap
y0 = min(bbox3.y0, bbox4.y0) - 0.05
y1 = max(bbox3.y1, bbox4.y1) + 0.05

line = plt.Line2D([x, x], [y0, y1], transform=fig.transFigure, linewidth=2)
fig.add_artist(line)
plt.savefig('figs/fig2.png', bbox_inches='tight')
plt.show()
