import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import torch
import math
from tqdm import tqdm
from scipy.stats import norm
import matplotlib.gridspec as gridspec
import matplotlib

def get_residual_corr(q, d, data):
    n = len(d)
    m = len(q)
    mu = q[None, :] - d[:, None]
    P = norm.cdf(mu)                            # fitted probs

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
        R = C / denom
        R[denom == 0] = 0.0
    return R


data = pd.read_csv('data/helm/helm_ans.csv').drop(columns=['question_id'])

openai_cols = ['openai/gpt-4-0613',
       'openai/gpt-4-1106-preview', 'openai/gpt-4-turbo-2024-04-09',
       'openai/gpt-4o-2024-05-13', 'openai/gpt-4o-2024-08-06']
second_cols = ['anthropic/claude-3-5-sonnet-20240620',
       'anthropic/claude-3-5-sonnet-20241022',
       'anthropic/claude-3-haiku-20240307',
       'anthropic/claude-3-opus-20240229', 'writer/palmyra-x-004', 'writer/palmyra-x-v3']
third_cols = ['meta/llama-3.1-8b-instruct-turbo', 'meta/llama-3.2-11b-vision-instruct-turbo', 'allenai/olmo-1.7-7b', 'allenai/olmo-7b',
'cohere/command-r', 'cohere/command-r-plus']

all_cols = data.columns

df = pd.read_csv('results/relativity_population/acsincome_rf_highly_correlated_logistic_small_mlp_d0.csv')

# This CSV already holds only the 20 highly correlated RFs (dt_cols[:10] + dt_cols[20:30]
# in scripts/experiments/population_relativity.py), followed by LR, MLP, and q columns.
rf_cols = [i for i in df.columns if 'randomforestclassifier' in i]
logistic_cols = [i for i in df.columns if 'logisticregression' in i]
mlp_cols = [i for i in df.columns if 'mlpclassifier' in i]
q_cols = [i for i in df.columns if 'q' in i]



matplotlib.rcParams['font.size'] = 16

# -----------------------
# Inputs (your existing)
# -----------------------
helm_files  = ['results/relativity_population/helm_exp1.pkl',
               'results/relativity_population/helm_exp2.pkl',
               'results/relativity_population/helm_exp3.pkl',
               'results/helm/helm_latent_p10.pkl']
helm_titles = ['OpenAI only',
               'OpenAI +\nAnthropic/Writer',
               'OpenAI +\nAnthropic/Writer +\nMeta/AllenAI/Cohere',
               'All Models']
helm_cols = [openai_cols, openai_cols + second_cols, openai_cols + second_cols + third_cols, all_cols]

rf_files  = ['acsincome_only_rf_highly_correlated',
             'acsincome_rf_highly_correlated_logistic_small',
             'acsincome_rf_highly_correlated_logistic_small_mlp',
             'acsincome_rf_highly_correlated_logistic_small_mlp_d0']
rf_titles = ['RF only', 'RF + LR', 'RF + LR + MLP', 'RF + LR +\nMLP + d0']
rf_cols_list = [rf_cols, rf_cols + logistic_cols, rf_cols + logistic_cols + mlp_cols,
            rf_cols + logistic_cols + mlp_cols + q_cols]

# -----------------------
# Figure layout: 2 rows x (8 plots + 1 colorbar col)
#   Row 0: heatmaps (tall)
#   Row 1: histograms (short)
# -----------------------
fig = plt.figure(figsize=(30, 6))
gs = gridspec.GridSpec(
    2, 9,
    width_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 0.05],
    height_ratios=[1.0, 0.25],   # make histogram row short
    wspace=0.1,
    hspace=0.01
)

axes_top  = [fig.add_subplot(gs[0, i]) for i in range(8)]
axes_hist = [fig.add_subplot(gs[1, i]) for i in range(8)]

# One colorbar axis spanning both rows
cbar_ax = fig.add_subplot(gs[:, 8])

# Shared colormap + normalization
cmap = "RdBu_r"
vmin, vmax = -1, 1
center = 0

# Convenience: store d’s (for hist row)
ds = []

# -----------------------
# Top row: first 4 (HELM)
# -----------------------
for i, (file, title, cols) in enumerate(zip(helm_files, helm_titles, helm_cols)):
    with open(file, "rb") as f:
        params = pickle.load(f)
    q = params['q']
    d = params["d"]
    Sigma = get_residual_corr(q, d, data[cols].to_numpy())
    ds.append(d)

    if i < 3:
        subset = Sigma[:5, :5]
    else:
        subset = Sigma[54:59, 54:59]

    # set diagonal to NaN
    np.fill_diagonal(subset, 1)

    # mask_upper = np.triu(np.ones_like(subset, dtype=bool), k=1)
    # subset[mask_upper] = np.nan
    sns.heatmap(
        subset,
        cmap=cmap, center=center, vmin=vmin, vmax=vmax,
        square=True,
        ax=axes_top[i],
        cbar=False
    )
    axes_top[i].set_title(title, fontsize=22)
    axes_top[i].set_xticks([])
    axes_top[i].set_yticks([])
    axes_top[i].set_xlabel("")
    axes_top[i].set_ylabel("")

# -----------------------
# Top row: next 4 (RF)
# -----------------------
for j, (file, title, cols) in enumerate(zip(rf_files, rf_titles, rf_cols_list), start=4):
    with open(f"results/relativity_population/{file}.pkl", "rb") as f:
        params = pickle.load(f)
    # Sigma = sigma_from_embeddings_np(params["v"])
    q = params['q']
    d = params["d"]
    # print(q.shape, d.shape)
    ds.append(d)
    Sigma = get_residual_corr(q, d, df[cols].astype(float).to_numpy())
    # set diagonal to NaN
    np.fill_diagonal(Sigma, 1)

    # mask_upper = np.triu(np.ones_like(Sigma, dtype=bool), k=1)
    # Sigma[mask_upper] = np.nan
    sns.heatmap(
        Sigma[:10, :10],
        cmap=cmap, center=center, vmin=vmin, vmax=vmax,
        square=True,
        ax=axes_top[j],
        cbar=False
    )
    axes_top[j].set_title(title, fontsize=22)
    axes_top[j].set_xticks([])
    axes_top[j].set_yticks([])
    axes_top[j].set_xlabel("")
    axes_top[j].set_ylabel("")

for ax in axes_top:
    ax.tick_params(bottom=False, labelbottom=False)

# -----------------------
# Panel labels (a) and (b)
# -----------------------
axes_top[0].text(
    0.02, 1.3, "(a) HELM",
    transform=axes_top[0].transAxes,
    ha="left", va="top",
    fontsize=22, fontweight="bold"
)
axes_top[4].text(
    0.02, 1.3, "(b) ACSIncome",
    transform=axes_top[4].transAxes,
    ha="left", va="top",
    fontsize=22, fontweight="bold"
)

# -----------------------
# Bottom row: histograms of d (short)
# -----------------------
def _as_1d_array(x):
    # robust: scalar -> [scalar], list/np -> flattened array
    try:
        arr = np.asarray(x).ravel()
        if arr.size == 0:
            return np.array([np.nan])
        return arr
    except Exception:
        return np.array([x])

for i in range(8):
    d_arr = _as_1d_array(ds[i])

    # If it's basically scalar, keep bins tight; otherwise use a reasonable default
    if d_arr.size <= 2 or np.nanmin(d_arr) == np.nanmax(d_arr):
        # single spike-ish
        axes_hist[i].hist(d_arr[~np.isnan(d_arr)], bins=1)
    else:
        axes_hist[i].hist(d_arr[~np.isnan(d_arr)], bins=25)

    axes_hist[i].set_yticks([])
    axes_hist[i].tick_params(axis="y", length=0)
    # keep x-axis minimal; comment out if you want tick labels
    # axes_hist[i].set_xticks([])
    axes_hist[i].set_xlabel("")  # no per-panel xlabel by default
    axes_hist[i].set_ylabel("")
    for spine in ["top", "right", "left"]:
        axes_hist[i].spines[spine].set_visible(False)

# Optional: one shared xlabel across the whole figure (covers both rows nicely)
fig.subplots_adjust(bottom=0.14)
fig.text(0.3, 0.3, r'Inferred excess correlation $\Sigma$', ha='center', va='bottom', fontsize=20)
fig.text(0.7, 0.3, r'Inferred excess correlation $\Sigma$', ha='center', va='bottom', fontsize=20)
fig.text(0.3, 0.025, r'Inferred difficulty $d$', ha='center', va='bottom', fontsize=20)
fig.text(0.7, 0.025, r'Inferred difficulty $d$', ha='center', va='bottom', fontsize=20)


# -----------------------
# One shared colorbar
# -----------------------
norm1 = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
sm = matplotlib.cm.ScalarMappable(norm=norm1, cmap=cmap)
sm.set_array([])
cb = fig.colorbar(sm, cax=cbar_ax)
# Enlarge tick labels
cb.ax.tick_params(
    labelsize=18,   # tick font size
    width=1.5,
    length=6
)
# cb.ax.set_ylabel("Correlation", fontsize=20)
cb.ax.set_title(
    "Correlation",
    fontsize=20,
    pad=15
)

from matplotlib.lines import Line2D

# Right edge of 4th subplot (index 3)
pos4 = axes_top[3].get_position()
pos5 = axes_top[4].get_position()

# x position halfway between 4th and 5th
x = 0.5 * (pos4.x1 + pos5.x0)

# vertical extent: from bottom of histogram row to top of heatmap row
y0 = axes_hist[0].get_position().y0
y1 = axes_top[0].get_position().y1

line = Line2D(
    [x, x],
    [y0, y1+0.1],
    transform=fig.transFigure,
    color="black",
    linewidth=1.5,
    alpha=0.8
)

fig.add_artist(line)

plt.savefig('figs/fig3.png', bbox_inches='tight')
plt.show()
