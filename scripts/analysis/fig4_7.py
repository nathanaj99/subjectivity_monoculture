import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import matplotlib

matplotlib.rcParams['font.size'] = 14


k_list = [2, 4, 8, 16, 32, 36, 40, 44, 48, 52, 56, 60, 64]


data = pd.read_csv('data/helm/helm_ans.csv').drop(columns=['question_id']).to_numpy()
results = pd.DataFrame(columns=['k', 'mse', 'off_inf', 'off_frob', 'off_median', 'off_mean'])

n, m = data.shape

with open('results/helm/helm_irt_all_k.pkl', 'rb') as f:
    params_all_k = pickle.load(f)

for k in k_list:
    params = params_all_k[k]

    A_hat = params['A']
    b_hat = params['b']
    Theta_hat = params['Theta']
    mu = b_hat[:, None] + A_hat @ Theta_hat.T   # shape (n, m)
    P = norm.cdf(mu)

    mse = np.sum((P - data)**2)/(data.shape[0]*data.shape[1])

    # --- Raw residuals (no Pearson scaling) ---
    E = data - P                                # (n, m)

    # Debias by removing column means (across items i)
    E_centered = E - E.mean(axis=0, keepdims=True)

    n = E.shape[0]

    # Raw residual covariance (population-style, divide by n)
    C = (E_centered.T @ E_centered) / n         # (m, m), set diag to 0 later if desired

    np.fill_diagonal(C, 0.0)
    tri = np.triu_indices(m, k=1)
    off_inf = np.max(np.abs(C[tri]))
    off_frob = np.sqrt((C[tri]**2).sum())
    results.loc[len(results)] = [k, mse, off_inf, off_frob, np.median(np.abs(C[tri])), np.abs(C[tri]).mean()]



data = pd.read_csv('data/hf/hf_ans.csv').drop(columns=['question_id']).to_numpy()
n, m = data.shape

results1 = pd.DataFrame(columns=['k', 'mse', 'off_inf', 'off_frob', 'off_median', 'off_mean'])
with open(f'results/hf/hf_irt_all_k.pkl', 'rb') as f:
    params = pickle.load(f)

for k in params.keys():
    A_hat = params[k]['A']
    b_hat = params[k]['b']
    Theta_hat = params[k]['Theta']

    mu = b_hat[:, None] + A_hat @ Theta_hat.T   # shape (n, m)
    P = norm.cdf(mu)

    mse = np.sum((P - data)**2)/(data.shape[0]*data.shape[1])

    E = data - P                                # (n, m)

    # Debias by removing column means (across items i)
    E_centered = E - E.mean(axis=0, keepdims=True)

    n = E.shape[0]

    # Raw residual covariance (population-style, divide by n)
    C = (E_centered.T @ E_centered) / n         # (m, m), set diag to 0 later if desired
    np.fill_diagonal(C, 0.0)
    tri = np.triu_indices(m, k=1)
    off_inf = np.max(np.abs(C[tri]))
    off_frob = np.sqrt((C[tri]**2).sum())
    results1.loc[len(results1)] = [k, mse, off_inf, off_frob, np.median(np.abs(C[tri])), np.abs(C[tri]).mean()]



##### PLOTTING FIG 7 #####
fig, ax = plt.subplots(2, 5, figsize=(20, 5), sharex=True)
import matplotlib
matplotlib.rcParams['font.size'] = 16
for j, result in enumerate([results, results1]):
    for i, (y, title) in enumerate(zip(['mse', 'off_inf', 'off_median', 'off_mean', 'off_frob'], ['MSE', 'AbsMax', 'AbsMedian', 'AbsMean', 'Frobenius Norm'])):
        sns.lineplot(data=result, x='k', y=y, ax=ax[j, i], marker='o')
        if j == 0:
            ax[j, i].set_title(title, fontsize=16)
        ax[j, i].set_ylabel('')
        if i == 2 and j == 1:
            ax[j, i].set_xlabel('K (Dimensions in MIRT)', fontsize=16)
        else:
            ax[j, i].set_xlabel('')
        ax[j, i].set_yscale('log')

    ax[j, 0].set_ylabel('HELM\n(MMLU)' if j == 0 else 'HF',
                        fontsize=16, fontweight='bold', rotation=0, labelpad=40)

fig.subplots_adjust(wspace=0.2, hspace=0.1)
plt.savefig('figs/fig7.png', bbox_inches='tight')
plt.show()




##### PLOTTING FIG 4 (Abridged version of Fig 7) #####
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

matplotlib.rcParams['font.size'] = 16

fig = plt.figure(figsize=(20,2.5))
gs = GridSpec(
    1, 5,
    width_ratios=[1, 1, 0.08, 1, 1],  # <-- spacer in the middle
    wspace=0.25
)

ax = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[0, 3]),
    fig.add_subplot(gs[0, 4]),
]

metrics = ['mse', 'off_mean']
titles = ['MSE', 'AbsMean']

# ---- HELM ----
for i, (y, title) in enumerate(zip(metrics, titles)):
    sns.lineplot(data=results, x='k', y=y, ax=ax[i], marker='o')
    ax[i].set_title(title)
    ax[i].set_yscale('log')
    ax[i].set_xlabel('')
    ax[i].set_ylabel('')

# ---- HF ----
for i, (y, title) in enumerate(zip(metrics, titles)):
    sns.lineplot(data=results1, x='k', y=y, ax=ax[i + 2], marker='o')
    ax[i + 2].set_title(title)
    ax[i + 2].set_yscale('log')
    ax[i + 2].set_xlabel('')
    ax[i + 2].set_ylabel('')

# Shared x-label
ax[1].set_xlabel('K (Dimensions in MIRT)', fontsize=16)
ax[2].set_xlabel('K (Dimensions in MIRT)', fontsize=16)

# Group labels
ax[0].set_ylabel(
    'HELM',
    fontsize=16,
    fontweight='bold',
    rotation=0,
    labelpad=30
)

ax[2].set_ylabel(
    'HF',
    fontsize=16,
    fontweight='bold',
    rotation=0,
    labelpad=15
)

plt.savefig('figs/fig4.png', bbox_inches='tight')
plt.show()
