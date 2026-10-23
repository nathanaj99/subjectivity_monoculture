import numpy as np
import torch
import os
import argparse
import pickle
import time
from tqdm import tqdm
import pandas as pd
import gc

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=500)
parser.add_argument('--m', type=int, default=50)
parser.add_argument('--num_covariates', type=int, default=5)
parser.add_argument('--lr', type=float, default=5e-2)
parser.add_argument('--max_iter', type=int, default=1000)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--k', type=int, default=2)
args = parser.parse_args()

n = args.n
m = args.m
num_covariates = args.num_covariates    
lr = args.lr
max_iter = args.max_iter
patience = args.patience
k = args.k
def compute_pairwise_sq_dists(X, w):
    """
    Compute pairwise squared Euclidean distances for rows of X.
    """
    diff = X[:, None, :] - X[None, :, :]
    return np.sum(w * diff ** 2, axis=2)

def simulate_covariates(m, num_covariates, seed, w):
    rng = np.random.default_rng(seed)
    x = rng.multivariate_normal(mean=np.zeros(num_covariates), cov=np.eye(num_covariates), size=m)
    pairwise_sq = compute_pairwise_sq_dists(np.asarray(x), w)
    K = np.exp(-pairwise_sq)
    Sigma = K + 1e-6*np.eye(m)
    
    return x, Sigma

# ----------------------
# 1) SIMULATION FUNCTION
# ----------------------
def simulate_data(n, m, d, q, Sigma, seed=None):
    """
    Simulate binary correctness data Y of shape (n, m) given:
      d: array of length n (difficulties)
      q: array of length m (abilities)
      Sigma: m x m covariance matrix (diag=1)
    Returns: Y (n x m), Z latent (n x m), epsilon (n x m)
    """
    if seed is not None:
        np.random.seed(seed)
    # simulate latent errors and data
    epsilon = np.random.multivariate_normal(mean=np.zeros(m), cov=Sigma, size=n)
    # latent scores
    Z = (q[np.newaxis, :] - d[:, np.newaxis]) + epsilon
    # binary outcomes
    Y = (Z > 0).astype(int)
    return Y, Z, epsilon

# # --- negative log‐posterior under probit 1PL ---
# def neg_probit1pl(params):
#     """
#     params = [d_0,...,d_{n_items-1},   q_0,...,q_{n_llms-1} ]
#     flat priors on d_i; q_j ~ Normal(0,1).
#     """
#     # unpack
#     d = params[             : n]
#     q = params[n      : n + m]

#     # log‐likelihood
#     ll = 0.0
#     for i in range(n):
#         # linear predictor for all models on item i
#         mu_ij = q - d[i]                       # shape (m,)
#         p_ij  = norm.cdf(mu_ij)                # Φ(q_j - d_i)
#         p_ij  = np.clip(p_ij, 1e-10, 1-1e-10)

#         y_ij = Y[i]                    # successes per LLM
#         ll  += np.sum(y_ij * np.log(p_ij) 
#                     + (m - y_ij) * np.log(1 - p_ij))

#     # Gaussian prior on q for location identifiability
#     ll += -0.5 * np.sum(q**2)

#     return -ll  # we minimize

from torch.distributions import Normal

def neg_probit_mirt_batched(
    Y, k=3, lr=1e-2, max_iter=1000, patience=20,
    batch_size_items=256, l2_theta=1.0, l2_A=1.0, l2_b=1e-2,
    whiten_every=25, whiten_warmup=10, eps=1e-6, verbose=True, seed=0
):
    """
    Probit MIRT (compensatory kD 2PL):
      mu_ij = b_i + A_i^T Theta_j,   P_ij = Phi(mu_ij).
    Missing Y allowed via NaNs.
    """
    # g = torch.Generator().manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator(device=device).manual_seed(seed)

    Y_t  = torch.as_tensor(Y, dtype=torch.float32, device=device)
    mask = ~torch.isnan(Y_t)
    Yobs = torch.nan_to_num(Y_t, nan=0.0)
    n, m = Yobs.shape

    # --- initialization: small random noise for A, Theta; b ~ 0 ---
    # def randn(*shape, scale=0.01):
    #     return torch.randn(*shape, generator=g, device=device) * scale

   # Good: leaf tensors (no grad_fn)
    Theta = torch.empty(m, k, device=device).normal_(mean=0.0, std=0.05).requires_grad_(True)
    A     = torch.empty(n, k, device=device).normal_(mean=0.0, std=0.05).requires_grad_(True)
    b     = torch.zeros(n, device=device, requires_grad=True)

    opt = torch.optim.Adam([{"params": [Theta, A], "lr": lr},
                            {"params": [b],        "lr": lr}], betas=(0.9, 0.999))
    stdN = Normal(0.0, 1.0)

    best, wait = float("inf"), 0

    def step_epoch():
        opt.zero_grad()
        total_ll = 0.0
        for i0 in range(0, n, batch_size_items):
            i1 = min(n, i0 + batch_size_items)
            Ai = A[i0:i1, :]                    # (B,k)
            bi = b[i0:i1]                       # (B,)
            mu = bi.unsqueeze(1) + Ai @ Theta.t()   # (B,m)
            P  = stdN.cdf(mu).clamp(eps, 1 - eps)
            Yi = Yobs[i0:i1, :]
            Mi = mask[i0:i1, :]
            ll = (Mi * (Yi * torch.log(P) + (1 - Yi) * torch.log(1 - P))).sum()
            (-ll).backward()
            total_ll += ll.item()
        # add L2 priors in the loss (lets autograd do the work)
        reg = 0.0
        if l2_theta > 0: reg = reg + 0.5 * l2_theta * (Theta.pow(2).sum())
        if l2_A     > 0: reg = reg + 0.5 * l2_A     * (A.pow(2).sum())
        if l2_b     > 0: reg = reg + 0.5 * l2_b     * (b.pow(2).sum())
        if reg != 0:
            reg.backward()
        return -(total_ll) + (reg.item() if isinstance(reg, torch.Tensor) else reg)

    # --- whitening (no reassignments) ---
    @torch.no_grad()
    def whiten_identifiability():
        # center Theta
        Theta.sub_(Theta.mean(dim=0, keepdim=True))
        # cov and whitening matrix W
        cov = (Theta.t() @ Theta) / float(m)
        evals, evecs = torch.linalg.eigh(cov)
        evals = torch.clamp(evals, min=1e-8)
        W = evecs @ torch.diag_embed(1.0 / torch.sqrt(evals)) @ evecs.t()
        # in-place updates keep leaf status
        Theta.copy_(Theta @ W)
        A.copy_(A @ torch.linalg.inv(W.t()))

    for epoch in tqdm(range(max_iter), disable=not verbose, desc="MIRT-probit"):
        negpost = step_epoch()
        opt.step()

        # optional whitening after some warmup, and not every epoch
        if whiten_every and (epoch + 1) >= whiten_warmup and ((epoch + 1) % whiten_every == 0):
            whiten_identifiability()

        # monitor & early stop
        if negpost < best - 1e-6:
            best, wait = negpost, 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"Early stopping @ {epoch}")
                break

        if verbose and (epoch % 50 == 0):
            with torch.no_grad():
                gA = 0.0 if A.grad is None else A.grad.norm().item()
                gT = 0.0 if Theta.grad is None else Theta.grad.norm().item()
                gb = 0.0 if b.grad is None else b.grad.norm().item()
                print(f"Epoch {epoch:03d}  neg.post={negpost:.3f}  |grad| A={gA:.3e} Θ={gT:.3e} b={gb:.3e}")

    A_hat     = A.detach().cpu().numpy()
    b_hat     = b.detach().cpu().numpy()
    Theta_hat = Theta.detach().cpu().numpy()

    del Y_t, Yobs, mask, A, b, Theta, opt, stdN
    torch.cuda.empty_cache(); gc.collect()
    return A_hat, b_hat, Theta_hat


def main(data_path, fp_out):
    Y = pd.read_csv(data_path)
    Y = Y.drop(columns=['question_id']).to_numpy().astype(float)
    n = Y.shape[0]
    m = Y.shape[1]
    if os.path.exists(fp_out):
        with open(fp_out, 'rb') as f:
            results_overall = pickle.load(f)
    else:
        results_overall = {}

    
    for k in [2, 4, 8, 16, 32, 36, 40, 44, 48, 52, 56, 60, 64]:
        if k in results_overall:
            continue
        A_hat, b_hat, Theta_hat = neg_probit_mirt_batched(Y, k=k, lr=lr, max_iter=max_iter, patience=900)
        results_overall[k] = {'A': A_hat, 'b': b_hat, 'Theta': Theta_hat}
        with open(fp_out, 'wb') as f:
            pickle.dump(results_overall, f)


def main_one_k(data_path, fp_out, k):
    Y = pd.read_csv(data_path)
    Y = Y.drop(columns=['question_id']).to_numpy().astype(float)
    n = Y.shape[0]
    m = Y.shape[1]
    A_hat, b_hat, Theta_hat = neg_probit_mirt_batched(Y, k=k, lr=lr, max_iter=max_iter, patience=900)
    results = {'A': A_hat, 'b': b_hat, 'Theta': Theta_hat}
    with open(fp_out, 'wb') as f:
        pickle.dump(results, f)
    return results

if __name__ == "__main__":
    main(data_path='data/helm/helm_ans.csv', fp_out='results/helm/helm_irt_all_k.pkl')
    main(data_path='data/hf/hf_ans.csv', fp_out='results/hf/hf_irt_all_k.pkl')

    main_one_k(data_path='data/helm/helm_ans.csv', fp_out='results/helm/helm_k2_irt.pkl', k=2)
    main_one_k(data_path='data/helm/helm_ans.csv', fp_out='results/helm/helm_k1_irt.pkl', k=1)