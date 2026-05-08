import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

k = 2
with open(f'results/helm/helm_k{k}_irt.pkl', 'rb') as f:
    params = pickle.load(f)
A_hat = params['A']
b_hat = params['b']
Theta_hat = params['Theta']

helm_data = pd.read_csv('data/helm/helm_ans.csv')
helm_data['question_type'] = helm_data['question_id'].apply(lambda x: '_'.join(x.split('_')[:-1]))

mean_acc = helm_data.drop(columns=['question_id']).groupby('question_type').mean()
Y = helm_data.drop(columns=['question_id', 'question_type']).to_numpy()

question_type = helm_data['question_type'].tolist()
models = helm_data.columns[1:-1]
model_accuracies = helm_data.drop(columns=['question_id', 'question_type']).mean().tolist()
company_models = [i.split('/')[0] for i in models]


Theta = np.asarray(Theta_hat, dtype=float)
acc = np.asarray(model_accuracies, dtype=float)
comp = np.asarray(company_models)

# --- clean mask: keep only finite rows across all inputs
mask = np.isfinite(Theta).all(axis=1) & np.isfinite(acc) & (comp != None)
Theta = Theta[mask]
acc = acc[mask]
comp = comp[mask]

x, y = Theta[:, 1], Theta[:, 0]

### HUGGINGFACE ###
k = 2
with open(f'results/hf/hf_irt_all_k.pkl', 'rb') as f:
    params = pickle.load(f)
A_hat1 = params[k]['A']
b_hat1 = params[k]['b']
Theta_hat1 = params[k]['Theta']

hf_data = pd.read_csv('data/hf/hf_ans.csv')
# Y = hf_data.drop(columns=['question_id', 'question_type']).to_numpy()
# question_type = hf_data['question_type'].tolist()
# models = hf_data.columns[1:-1]
model_accuracies1 = hf_data.drop(columns=['question_id']).mean().tolist()
# company_models = [i.split('/')[0] for i in models]


Theta1 = np.asarray(Theta_hat1, dtype=float)
acc1 = np.asarray(model_accuracies1, dtype=float)
# comp = np.asarray(company_models)

# --- clean mask: keep only finite rows across all inputs
mask = np.isfinite(Theta1).all(axis=1) & np.isfinite(acc1)
Theta1 = Theta1[mask]
acc1 = acc1[mask]
# comp = comp[mask]

x1, y1 = Theta1[:, 0], Theta1[:, 1]

merged = pd.read_csv('data/hf/hf_ans_with_metadata.csv')

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
mean_acc = merged.drop(columns=['question_id', 'doc', 'doc_hash', 'src']).groupby('category').mean()
# mean_acc: rows = question types, columns = models
X = mean_acc.values

# Standardize across question types
X_std = StandardScaler().fit_transform(X)

# PCA
pca1 = PCA()
pca1.fit(X_std)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


fig, (ax00, ax10, ax12) = plt.subplots(
    1, 3, figsize=(18, 2.5),
    gridspec_kw=dict(wspace=0.28)
)

# =======================
# (a) HELM – Ability space
# =======================
sc0 = ax00.scatter(
    x, y, c=acc, cmap='viridis',
    s=40, edgecolors='k', alpha=0.85
)
ax00.axhline(0, lw=1, ls='--', color='gray')
ax00.axvline(0, lw=1, ls='--', color='gray')
ax00.set_ylabel(r"$\Theta_{:,2}$", fontsize=16)
ax00.set_xlabel(r"$\Theta_{:,1}$", fontsize=16)
ax00.set_xlim(-2, 2)
ax00.set_ylim(-2, 2)
ax00.set_title("Models in Ability Space")

cbar0 = fig.colorbar(sc0, ax=ax00, fraction=0.046, pad=0.04)
cbar0.ax.set_xlabel("Acc.", labelpad=5)
cbar0.ax.xaxis.set_label_position("top")

ax00.text(
    -0.2, 1.1, "(a) HELM",
    transform=ax00.transAxes,
    fontsize=16, fontweight="bold"
)

# =======================
# (b) HF – Ability space
# =======================
sc1 = ax10.scatter(
    x1, y1, c=acc1, cmap='viridis',
    s=40, edgecolors='k', alpha=0.85
)
ax10.axhline(0, lw=1, ls='--', color='gray')
ax10.axvline(0, lw=1, ls='--', color='gray')
ax10.set_xlabel(r"$\Theta_{:,1}$", fontsize=16)
# ax10.set_ylabel(r"$\Theta_{:,2}$")
ax10.set_title("Models in Ability Space")

cbar1 = fig.colorbar(sc1, ax=ax10, fraction=0.046, pad=0.04)
cbar1.ax.set_xlabel("Acc.", labelpad=5)
cbar1.ax.xaxis.set_label_position("top")

ax10.text(
    -0.15, 1.1, "(b) HF",
    transform=ax10.transAxes,
    fontsize=16, fontweight="bold"
)

# =======================
# (c) HF – PCA scree plot
# =======================
ax12.plot(
    range(1, 11),
    pca1.explained_variance_ratio_[:10],
    marker='o'
)
ax12.set_xlabel("Principal Component", fontsize=16)
ax12.set_ylabel("Variance Explained", fontsize=16)
ax12.set_title("PCA on Accuracy by Question Type")


# Get subplot 1 (ax00) position in figure coordinates
pos = ax00.get_position()

x0_coord, y0_coord = pos.x0, pos.y0
x1_coord, y1_coord = pos.x1, pos.y1

# Horizontal line at bottom of subplot 1
hline = Line2D(
    [x0_coord-0.03, x1_coord+0.045], [y0_coord-0.23, y0_coord-0.23],
    transform=fig.transFigure,
    color="black",
    linewidth=1.5
)

# Vertical line at right of subplot 1
vline = Line2D(
    [x1_coord+0.045, x1_coord+0.045], [y0_coord-0.23, y1_coord+0.15],
    transform=fig.transFigure,
    color="black",
    linewidth=1.5
)

fig.add_artist(hline)
fig.add_artist(vline)
# wspace = 0.35
# plt.subplots_adjust(wspace=0.01)
# ax12.text(
#     -0.15, 1.1, "(c)",
#     transform=ax12.transAxes,
#     fontsize=16, fontweight="bold"
# )

plt.savefig("figs/fig5_top.png", bbox_inches='tight')
plt.show()


#### PLOT BOTTOM ####
company_list = [i.split('/')[0] for i in hf_data.columns[1:]]
base_model_list = [i.split('/')[1].split('-')[0] for i in hf_data.columns[1:]]

comp = np.asarray(company_list)

# Example: define which companies go in which subplot
# (replace these with your actual lists)
companies_fig1 = ['jeffmeloy', 'lemon07r', 'qingy2024']
companies_fig2 = ['newsbang', 'Qwen', 'sometimesanotion', 'T145']
companies_fig3 = ['bunnycore', 'dnhkng', 'DreadPoor', 'MaziyarPanahi']
fig_companies = [companies_fig1, companies_fig2, companies_fig3]

x, y = Theta1[:, 0], Theta1[:, 1]

# --- global style map (unique marker per company across all subplots)
color_cycle  = plt.cm.tab10.colors  # 10 colors
marker_cycle = ['o','s','^','D','P','X','v','<','>','H','*','p','h','8','d','1','2','3','4','|','_']

# all companies that appear in any panel AND are in your data
all_companies = sorted(
    set(companies_fig1) | set(companies_fig2) | set(companies_fig3)
)
all_companies = [c for c in all_companies if c in comp]

if len(all_companies) > len(marker_cycle):
    print("Warning: more companies than markers, some markers will repeat.")

style_map = {}
legend_handles = []

# We'll create dummy handles for the shared legend as we assign styles
for i, c in enumerate(all_companies):
    color  = color_cycle[i % len(color_cycle)]
    marker = marker_cycle[i % len(marker_cycle)]
    style_map[c] = dict(color=color, marker=marker)

# --- plotting in 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 3), sharex=True, sharey=True)

for ax, companies_for_ax in zip(axes, fig_companies):
    for c in companies_for_ax:
        if c not in style_map:
            continue
        idx = (comp == c)
        if not np.any(idx):
            continue

        style = style_map[c]

        sc = ax.scatter(
            x[idx], y[idx],
            s=80,
            edgecolors='k',
            alpha=0.9,
            **style,
        )

    ax.axhline(0, lw=1, ls='--', color='gray')
    ax.axvline(0, lw=1, ls='--', color='gray')
    ax.set_xlabel(r"$\Theta_{:,1}$", fontsize=16)

axes[0].set_ylabel(r"$\Theta_{:,2}$", fontsize=16)
# axes[0].set_title("Panel 1")
# axes[1].set_title("Panel 2")
# axes[2].set_title("Panel 3")

# --- build shared legend from the style_map
legend_handles = []
legend_labels  = []

for c, style in style_map.items():
    h = axes[0].scatter([], [], s=50, edgecolors='k', alpha=0.9,
                        **style, label=c)
    legend_handles.append(h)

# plt.legend(handles=legend_handles, title="Contributor", bbox_to_anchor=(1.02, 1.1), loc='upper left')

plt.legend(
    handles=legend_handles,
    title="Contributor",
    ncol=2,
    bbox_to_anchor=(1.02, 1.1),
    loc="upper left",
    columnspacing=1.2,
    handletextpad=0.6,
    fontsize=14
)
plt.tight_layout()
plt.subplots_adjust(wspace=0.05)

plt.savefig("figs/fig5_bottom.png", bbox_inches='tight')
plt.show()
