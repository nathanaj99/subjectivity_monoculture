import torch
from tqdm import tqdm
import numpy as np
import pickle
import pandas as pd
import time
import os
import argparse
import math
from torch.distributions import MultivariateNormal
import gc
from scipy.stats import norm
parser = argparse.ArgumentParser()
# parser.add_argument('--save_path', type=str, default='results_kernel_gpu.pkl')
# parser.add_argument('--data_path', type=str, default='data/huggingface/hf_ans.csv')
# parser.add_argument('--covariates_path', type=str, default='data/huggingface/hf_normalized.csv')
parser.add_argument('--max_iter', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--R', type=int, default=100)
# parser.add_argument('--seed', type=int, default=40)
parser.add_argument('--n', type=int, default=500)
parser.add_argument('--m', type=int, default=50)
parser.add_argument('--num_covariates', type=int, default=10)
parser.add_argument('--w', type=list, default=[0.5]*10)
parser.add_argument('--subsample_frac', type=float, default=1.0)
parser.add_argument('--k', type=int, default=20)
# parser.add_argument('--o', type=int, default=10)
# parser.add_argument('--p', type=float, default=0.1)
# parser.add_argument('--ell', type=float, default=2)
args = parser.parse_args()

# save_path = args.save_path
max_iter = args.max_iter
lr = args.lr
R = args.R
# data_path = args.data_path
# covariates_path = args.covariates_path
# seed = args.seed
n = args.n
m = args.m
num_covariates = args.num_covariates
w = args.w
subsample_frac = args.subsample_frac
k = args.k
# o = args.o
# p = args.p
# ell = args.ell

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

def simulate_unit_embeddings(m, k=2, seed=None, jitter=1e-6):
    """
    Draw V_j ~ N(0, I_k), project to the unit sphere, and
    return Sigma = V Vᵀ + jitter·I.

    Parameters
    ----------
    m : int            # number of models
    k : int, default 2 # latent dimension
    seed : int | None
    jitter : float     # diagonal nugget for numerical stability

    Returns
    -------
    V      : (m, k)  unit vectors
    Sigma  : (m, m)  correlation (covariance) matrix
    """
    rng = np.random.default_rng(seed)
    V = rng.standard_normal(size=(m, k))
    V /= np.linalg.norm(V, axis=1, keepdims=True)        # project to unit sphere
    Sigma = V @ V.T + jitter * np.eye(m)                 # Gram + nugget
    return V, Sigma

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

class TruncatedNormal:
    def __init__(self, mean, std, a, b):
        self.mean = mean
        self.std = std
        self.a = a
        self.b = b

        # Standardize bounds
        self.alpha = (a - mean) / std
        self.beta = (b - mean) / std

        # Standard normal CDF
        self.device = mean.device if isinstance(mean, torch.Tensor) else 'cpu'
        self.normal = torch.distributions.Normal(
            torch.tensor(0.0, device=self.device),
            torch.tensor(1.0, device=self.device)
        )

        self.Z = self.normal.cdf(self.beta) - self.normal.cdf(self.alpha)

    def sample(self):
        # Uniform sample in [CDF(alpha), CDF(beta)]
        u = torch.rand(self.mean.shape, device=self.device) * self.Z + self.normal.cdf(self.alpha)
        # Clamp u to avoid inf/-inf from icdf
        eps = torch.finfo(u.dtype).eps
        u = torch.clamp(u, eps, 1 - eps)
        # Inverse CDF (ppf)
        z = self.normal.icdf(u)
        return self.mean + self.std * z
    
def sample_trunc_mvnorm_ghk_pytorch(mu, Sigma, Y_i, R=100, device=None):
    """
    Vectorized GHK sampler for Z ~ N(mu, Sigma) truncated by signs Y_i.
    Samples R replicates in parallel on GPU.
    """
    device = device or mu.device
    m = mu.shape[0]
    # Cholesky factorization
    L = torch.linalg.cholesky(Sigma)
    # allocate latent gaussian in y-space
    y = torch.zeros(R, m, device=device)
    # sequentially sample each dimension j in y-space
    for j in range(m):
        # compute linear combination of previous y's: sum_k<j L[j,k] * y[:,k]
        if j > 0:
            prev = y[:, :j]
            weights = L[j, :j]
            mean_affine = (prev * weights).sum(dim=1)
        else:
            mean_affine = torch.zeros(R, device=device)
        # threshold in y-space
        mu_j = mu[j]
        ljj = L[j, j]
        loc = (-mu_j - mean_affine) / ljj
        # set truncation bounds for R replicas
        if Y_i[j] == 1:
            a = loc
            b = torch.full((R,), float('inf'), device=device)
        else:
            a = torch.full((R,), float('-inf'), device=device)
            b = loc

        tn = TruncatedNormal(
            torch.zeros(R, device=device),
            torch.ones(R, device=device),
            a, b
        )
        y[:, j] = tn.sample()
    # map back to original space: z = mu + y @ L.T
    Z = mu.unsqueeze(0) + y @ L.T
    return Z


def gradient_probit_torch(Y, d, q, Sigma, m_star, S_star):
    """
    Compute gradients wrt d, q, and Sigma in PyTorch.
    """
    n, m = Y.shape
    invS = torch.linalg.inv(Sigma).float()
    g_d = torch.zeros_like(d)
    g_q = torch.zeros_like(q)
    for i in range(n):
        mu_i = q - d[i]
        Delta = m_star[i] - mu_i
        g_d[i] = - (invS @ Delta).sum()
        g_q += invS @ Delta
    Sbar = S_star.mean(dim=0)
    g_Sigma = 0.5 * invS @ (Sbar - Sigma) @ invS
    return g_d, g_q, g_Sigma, Sbar

def _gauss_legendre(n=20, device=None, dtype=torch.float64):
    """Return nodes x_i ∈ (0,1) and weights w_i for ∫₀¹ f(t) dt."""
    x, w = np.polynomial.legendre.leggauss(n)          # on (−1,1)
    x = (x + 1.0) * 0.5                                # → (0,1)
    w = w * 0.5
    return (torch.as_tensor(x, device=device, dtype=dtype),
            torch.as_tensor(w, device=device, dtype=dtype))


def bi_cdf(a, b, rho, n_quad=20):
    """
    Vectorised bivariate normal CDF Φ₂(a,b;ρ).

    Parameters
    ----------
    a, b, rho : torch.Tensor
        Broadcast-compatible tensors.  No grad is needed; the function is
        pure numerics.
    n_quad : int
        # Gauss–Legendre nodes (default 20 gives ~1e-6 max error).

    Returns
    -------
    Φ₂(a,b;ρ) with the broadcast shape.
    """
    device = a.device
    dtype  = torch.float64          # keep high precision internally
    a  = a.to(dtype)
    b  = b.to(dtype)
    rho = torch.clamp(rho.to(dtype), -0.999999, 0.999999)

    # Univariate CDFs
    Phi   = torch.distributions.Normal(0.0, 1.0)
    Phi_a = Phi.cdf(a)
    Phi_b = Phi.cdf(b)

    # Gauss–Legendre nodes on (0,1)
    xg, wg = _gauss_legendre(n_quad, device=device, dtype=dtype)
    # Reshape nodes for broadcasting: (n_quad, 1, 1, …)
    while xg.dim() < a.dim() + 1:
        xg = xg.unsqueeze(-1)
        wg = wg.unsqueeze(-1)

    r_abs = rho.abs()
    t     = xg * r_abs.unsqueeze(0)                    # (n_quad, …)
    one_m_t2 = 1.0 - t.pow(2)
    expon    = torch.exp(-(a.pow(2) - 2*a*b*t + b.pow(2))
                         / (2.0 * one_m_t2))
    integrand = expon / torch.sqrt(one_m_t2)
    T = (wg * integrand).sum(0) * r_abs / (2.0 * math.pi)

    cdf = Phi_a * Phi_b + torch.sign(rho) * T
    return cdf.to(a.dtype)         # cast back to original dtype

def fit_probit_kernel(
    Y, X,
    R=100, lr=1e-3, max_iter=100,
    beta1=0.9, beta2=0.999, epsilon=1e-8,
    patience=10,
    device=None, save_path=None,
    # New parameters for continuing optimization
    initial_d=None, initial_q=None, initial_w=None,
    initial_m_w=None, initial_v_w=None,
    start_iter=0, history=None, loss=None
):
    """
    Fit probit-IRT with correlation-kernel covariance in PyTorch (GPU).
    Y : (n,m) binary response matrix
    X : (m,p) feature matrix for models
    Returns:
      d, q, history of (rho, ell)
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    Y = torch.as_tensor(Y, dtype=torch.int8, device=device)
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    n, m = Y.shape
    num_covariates = X.shape[1]

    # Initialize parameters
    d = torch.as_tensor(initial_d, device=device, dtype=torch.float32) if initial_d is not None else torch.zeros(n, device=device, dtype=torch.float32)
    q = torch.as_tensor(initial_q, device=device, dtype=torch.float32) if initial_q is not None else torch.zeros(m, device=device, dtype=torch.float32)
    w = torch.as_tensor(initial_w, device=device, dtype=torch.float32) if initial_w is not None else torch.rand(num_covariates, device=device, dtype=torch.float32)
    # v = torch.as_tensor(torch.log(w), device=device, dtype=torch.float32) # log-weights

    best_loss = float('inf') if loss is None else loss
    
    # Adam moments
    m_w = torch.as_tensor(initial_m_w, device=device) if initial_m_w is not None else torch.zeros_like(w).to(device)
    v_w = torch.as_tensor(initial_v_w, device=device) if initial_v_w is not None else torch.zeros_like(w).to(device)

    # precompute differences
    diff3 = (X[:, None, :] - X[None, :, :])  # shape (m,m,num_covariates)

    if start_iter == 0:
        assert history is None
        history = {'w': [w.detach().cpu().numpy()], 'loss': [float('inf')],
        'converged': False}
    else:
        assert history is not None and history['converged'] is False

    no_improve = 0

    # precompute all mu matrix: shape (n, m)
    mu = q.unsqueeze(0) - d.unsqueeze(1)  # then mu[i,j] = q_j - d_i
    
    # Stage 2 update: replace your moment-matching
    for it in tqdm(range(max_iter)):
        sq   = (diff3.pow(2) * w[None, None, :]).sum(2)          # r_{jk}
        K    = torch.exp(-sq)                   # ρ_{jk}
        Sigma = K + 1e-6 * torch.eye(m, device=device)

        #######################################################################
        # Moment-matching loss and manual gradients wrt w and ell
        #######################################################################
        with torch.no_grad():
            # ----- empirical second-order moments ---------------------------
            Yf     = Y.float()
            hat_m  = Yf.mean(0)                        # (m,)
            C_hat  = (Yf.t() @ Yf)/n - hat_m[:,None]*hat_m[None,:]  # (m,m)

        # ---------- model-predicted moments ---------------------------------
        Phi_mu   = torch.special.ndtr(mu)              # (n,m)
        m_pred   = Phi_mu.mean(0)                      # (m,)

        # pairwise bivariate CDF needed for C_model
        # bi_cdf(a,b,rho) must broadcast a,b (n,m,m) with rho (m,m)
        a = mu.unsqueeze(2)        # (n,m,1)
        b = mu.unsqueeze(1)        # (n,1,m)
        # After computing K  (ρ = K)
        eps_rho = 1e-6               # 1e-6 is plenty; adjust if needed
        rho     = torch.clamp(K, -1.0 + eps_rho, 1.0 - eps_rho)
        # rho = K                                                        # (m,m)
        p11 = bi_cdf(a, b, rho)                       # (n,m,m)
        C_model = p11.mean(0) - m_pred[:,None] * m_pred[None,:]  # (m,m)

        # ----- bivariate pdf \bar{phi}_{jk} (mean over items) ---------------
        one_minus_r2 = 1.0 - rho**2
        inv_sqrt     = (2*math.pi*torch.sqrt(one_minus_r2))
        expon        = torch.exp(-(a**2 - 2*rho*a*b + b**2) /
                                 (2*one_minus_r2))
        phi2         = expon / inv_sqrt                # (n,m,m)
        mean_phi2    = phi2.mean(0)                    # (m,m)
        # >>> mask out the diagonal so it never propagates <<<
        diag_mask = torch.eye(m, dtype=torch.bool, device=device) # (m,m)
        mean_phi2.masked_fill_(diag_mask, 0.0) # (m,m)

        # <<< ------------------------------------------------ >>

        # ---------- loss ----------------------------------------------------
        diff      = C_model - C_hat                    # (m,m)
        # diff.masked_fill_(diag_mask, 0.0)      # (m,m)
        loss_pair = (diff.triu(1)**2).sum()            # scalar, j<k only
        loss_val  = loss_pair.item()

        # ---------- manual gradients ----------------------------------------
        Delta = 2.0 * diff * mean_phi2                # (m,m)
        if torch.isnan(Delta).any():
            print("Warning: NaN values detected in Delta")

        # common factors ------------------------------------------------------
        # sq      = (diff3**2 * w[None, None, :]).sum(2)    # r_{jk}, (m,m)
        # grad_ell = (Delta * rho * sq / (ell**3)).triu(1).sum()

        # weight gradients ----------------------------------------------------
        # ∂L/∂w_d = -½ Σ_{j<k} Δ_{jk} * ρ_{jk} * δ_{jk,d}²
        grad_w = (-(Delta.unsqueeze(-1) * rho.unsqueeze(-1) * diff3.pow(2))
                  .sum((0,1)))                         # (p,)

        # Adam-style update on w and ell
        with torch.no_grad():
            # update w
            m_w = beta1*m_w + (1-beta1)*grad_w
            v_w = beta2*v_w + (1-beta2)*(grad_w**2)
            w -= lr * (m_w/(1-beta1**(it+1))) / (v_w/(1-beta2**(it+1))).sqrt().add(epsilon)
            w.clamp_(min=1e-6)
        
        # early stop
        if loss_val<best_loss: 
            best_loss=loss_val
            no_improve=0
        else: 
            no_improve+=1;
            if no_improve>=patience: 
                history['converged'] = True
                print(f"Early stopping at iter {it}: loss did not improve in {patience} steps.")
                break
        
        # Save history
        # history['d'].append(d.cpu().numpy())
        # history['q'].append(q.cpu().numpy())
        history['w'].append(w.detach().cpu().numpy())
        history['loss'].append(loss_val)

@torch.no_grad()
def full_kernel_metrics_tiled(V, true_v, model_block=128):
    """
    Compute RMSE and Pearson correlation between K_true=exp(-||v*_i - v*_j||^2)
    and K_est=exp(-||V_i - V_j||^2) over all i<j, without forming full m×m matrices.

    Returns: dict with rmse_K, corr_K. Also returns distance metrics rmse_D, corr_D.
    """
    device = V.device
    dtype_acc = torch.float64  # for stable sums

    def pairwise_sqdist_block(A, B):
        a2 = (A*A).sum(dim=1, keepdim=True)         # (b1,1)
        b2 = (B*B).sum(dim=1, keepdim=True).T       # (1,b2)
        D  = a2 + b2 - 2.0*(A @ B.T)
        return torch.clamp(D, min=0.0)

    m = V.shape[0]
    # Accumulators for kernel stats
    SxK = torch.zeros((), device=device, dtype=dtype_acc)  # sum K_true
    SyK = torch.zeros((), device=device, dtype=dtype_acc)  # sum K_est
    SxxK = torch.zeros((), device=device, dtype=dtype_acc) # sum K_true^2
    SyyK = torch.zeros((), device=device, dtype=dtype_acc) # sum K_est^2
    SxyK = torch.zeros((), device=device, dtype=dtype_acc) # sum K_true*K_est
    # Accumulators for distance stats (optional but cheap once we have blocks)
    SxD = torch.zeros((), device=device, dtype=dtype_acc)
    SyD = torch.zeros((), device=device, dtype=dtype_acc)
    SxxD = torch.zeros((), device=device, dtype=dtype_acc)
    SyyD = torch.zeros((), device=device, dtype=dtype_acc)
    SxyD = torch.zeros((), device=device, dtype=dtype_acc)

    Npairs = 0  # integer count of i<j

    for j0 in range(0, m, model_block):
        j1 = min(m, j0 + model_block)
        v1_true = true_v[j0:j1]
        v1_est  = V[j0:j1]

        for k0 in range(j0, m, model_block):  # upper-triangular tiles
            k1 = min(m, k0 + model_block)
            v2_true = true_v[k0:k1]
            v2_est  = V[k0:k1]

            D_true_blk = pairwise_sqdist_block(v1_true, v2_true)   # (b1,b2)
            D_est_blk  = pairwise_sqdist_block(v1_est,  v2_est)    # (b1,b2)
            K_true_blk = torch.exp(-D_true_blk)
            K_est_blk  = torch.exp(-D_est_blk)

            if j0 == k0:
                b = j1 - j0
                iu = torch.triu_indices(b, b, offset=1, device=device)
                # kernel
                Xt = K_true_blk[iu[0], iu[1]].to(dtype_acc)
                Yt = K_est_blk [iu[0], iu[1]].to(dtype_acc)
                # distance
                Xd = D_true_blk[iu[0], iu[1]].to(dtype_acc)
                Yd = D_est_blk [iu[0], iu[1]].to(dtype_acc)
                npairs_blk = b*(b-1)//2
            else:
                Xt = K_true_blk.reshape(-1).to(dtype_acc)
                Yt = K_est_blk .reshape(-1).to(dtype_acc)
                Xd = D_true_blk.reshape(-1).to(dtype_acc)
                Yd = D_est_blk .reshape(-1).to(dtype_acc)
                npairs_blk = (j1-j0)*(k1-k0)

            # accumulate kernel stats
            SxK  += Xt.sum()
            SyK  += Yt.sum()
            SxxK += (Xt*Xt).sum()
            SyyK += (Yt*Yt).sum()
            SxyK += (Xt*Yt).sum()
            # accumulate distance stats
            SxD  += Xd.sum()
            SyD  += Yd.sum()
            SxxD += (Xd*Xd).sum()
            SyyD += (Yd*Yd).sum()
            SxyD += (Xd*Yd).sum()

            Npairs += npairs_blk

            # free block tensors early
            del D_true_blk, D_est_blk, K_true_blk, K_est_blk, Xt, Yt, Xd, Yd

    N = float(Npairs)
    # Kernel metrics
    mean_xK = (SxK / N).item()
    mean_yK = (SyK / N).item()
    var_xK  = (SxxK / N - (SxK / N)**2).item()
    var_yK  = (SyyK / N - (SyK / N)**2).item()
    cov_xyK = (SxyK / N - (SxK / N)*(SyK / N)).item()
    rmse_K  = math.sqrt(max(0.0, ((SxxK + SyyK - 2*SxyK) / N).item()))
    denomK  = math.sqrt(max(0.0, var_xK) * max(0.0, var_yK))
    corr_K  = (cov_xyK / denomK) if denomK > 0 else float('nan')

    # Distance metrics
    mean_xD = (SxD / N).item()
    mean_yD = (SyD / N).item()
    var_xD  = (SxxD / N - (SxD / N)**2).item()
    var_yD  = (SyyD / N - (SyD / N)**2).item()
    cov_xyD = (SxyD / N - (SxD / N)*(SyD / N)).item()
    rmse_D  = math.sqrt(max(0.0, ((SxxD + SyyD - 2*SxyD) / N).item()))
    denomD  = math.sqrt(max(0.0, var_xD) * max(0.0, var_yD))
    corr_D  = (cov_xyD / denomD) if denomD > 0 else float('nan')

    return {
        'rmse_K': rmse_K, 'corr_K': corr_K,
        'rmse_D': rmse_D, 'corr_D': corr_D,
        'mean_K_true': mean_xK, 'mean_K_est': mean_yK,
        'mean_D_true': mean_xD, 'mean_D_est': mean_yD,
        'Npairs': Npairs,
    }


# ------------------------------------------------------------
#  Main optimisation loop
# ------------------------------------------------------------
def fit_kernel_latent_stream(
        Y,                 # (n,m) 0/1  (kept on CPU)
        V,                 # (m,k) learnable embeddings Γ (updated in-place on device)
        q, d,              # q: (m,), d: (n,)
        true_v=None,       # (m,k) optional, for diagnostics
        beta1=0.9, beta2=0.999, epsilon=1e-8,
        patience=10,
        item_chunk=256,    # items per sub-batch (B)
        model_block=128,   # models per tile (b)
        lr=1e-3, max_iter=200,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        history=None):

    """
    Minimal changes from your RBF version:
      - Uses dot-product correlations rho_{jk} = v_j^T v_k (no tau, no RBF).
      - Keeps streaming/tiled computation.
      - Projects each row v_j to ||v_j|| <= 1 after every Adam step
        to guarantee Σ = ΓΓ^T + diag(1 - ||v_j||^2) is PSD with unit diagonal.
    """

    # ------------------- tensorize (lean) -----------------------------------
    Y = torch.as_tensor(Y, dtype=torch.float32, device='cpu')
    q = torch.as_tensor(q, dtype=torch.float32, device=device)
    d = torch.as_tensor(d, dtype=torch.float32, device=device)
    V = torch.as_tensor(V, dtype=torch.float32, device=device)  # rows are γ_j

    n, m = Y.shape
    k = V.shape[1]
    use_cuda = (isinstance(device, str) and device.startswith('cuda')) or (
        hasattr(device, "type") and device.type == 'cuda'
    )

    std_norm = torch.distributions.Normal(0.0, 1.0)

    # ------------------- Pass 0: per-model means from data -------------------
    with torch.no_grad():
        Ybar = Y.mean(0)  # (m,) CPU float32

    # ------------------- Adam buffers ----------------------------------------
    m_V = torch.zeros_like(V)
    v_V = torch.zeros_like(V)

    # ------------------- history ---------------------------------------------
    history = history or {'loss': [], 'v_norm_max': [], 'rmse_K': [], 'corr_K': []}
    best_loss, no_improve = float('inf'), 0

    for it in tqdm(range(max_iter), desc="fit_kernel_latent_stream (dot-product tiled)"):
        # ---------------- Pass 1: m_pred = E[Y_hat] per model ----------------
        # S_phi[j] = sum_i Phi(q_j - d_i)
        S_phi = torch.zeros(m, dtype=torch.float32, device=device)
        for i0 in range(0, n, item_chunk):
            i1 = min(n, i0 + item_chunk)
            dd = d[i0:i1]  # (B,)
            for j0 in range(0, m, model_block):
                j1 = min(m, j0 + model_block)
                muJ = q[j0:j1].unsqueeze(0) - dd.unsqueeze(1)   # (B,b)
                Phi = std_norm.cdf(muJ)                         # (B,b)
                S_phi[j0:j1] += Phi.sum(0)
                del muJ, Phi
            if use_cuda:
                torch.cuda.empty_cache()
        m_pred = S_phi / n  # (m,)

        # ---------------- Pass 2: tile over (J,K); stream over items ---------
        loss_val = torch.zeros((), dtype=torch.float32, device=device)

        # Gradient accumulator: Z ≈ Σ_k G_{·k} v_k
        Z = torch.zeros_like(V)  # (m,k)

        for j0 in range(0, m, model_block):
            j1 = min(m, j0 + model_block)
            v1 = V[j0:j1]        # (b1,k) on device
            q1 = q[j0:j1]        # (b1,)
            mp1 = m_pred[j0:j1]  # (b1,)

            for k0 in range(j0, m, model_block):   # upper triangle
                k1 = min(m, k0 + model_block)
                v2 = V[k0:k1]        # (b2,k)
                q2 = q[k0:k1]
                mp2 = m_pred[k0:k1]

                b1, b2 = j1 - j0, k1 - k0

                # Blockwise empirical accumulators
                S_p11_blk    = torch.zeros((b1, b2), dtype=torch.float32, device=device)
                S_phibar_blk = torch.zeros((b1, b2), dtype=torch.float32, device=device)
                S_y11_blk    = torch.zeros((b1, b2), dtype=torch.float32, device='cpu')

                # -------- correlations by dot product (NO RBF, NO tau) -------
                # rho_blk: (b1,b2) with off-diagonal entries v1 @ v2^T
                rho_blk = (v1 @ v2.T).clamp(min=-0.999, max=0.999)  # stability

                # Zero out diagonal entries inside a diagonal block
                if j0 == k0:
                    rho_blk = rho_blk.clone()
                    rho_blk.fill_diagonal_(0.0)

                # -------- stream over items for this block --------
                for i2 in range(0, n, item_chunk):
                    i3 = min(n, i2 + item_chunk)
                    dd = d[i2:i3]                      # (B,)

                    a = q1.unsqueeze(0) - dd.unsqueeze(1)   # (B,b1)
                    b = q2.unsqueeze(0) - dd.unsqueeze(1)   # (B,b2)
                    A = a.unsqueeze(2)                      # (B,b1,1)
                    B = b.unsqueeze(1)                      # (B,1,b2)

                    # BVN CDF and PDF contributions at correlation rho_blk
                    p11 = bi_cdf(A, B, rho_blk)            # (B,b1,b2)

                    one_m_r2  = 1.0 - rho_blk * rho_blk    # (b1,b2)
                    inv_const = 1.0 / (2*math.pi*torch.sqrt(one_m_r2))
                    quad      = (A*A - 2.0*rho_blk*A*B + B*B) / (2.0*one_m_r2)
                    phibar    = torch.exp(-quad) * inv_const          # (B,b1,b2)

                    S_p11_blk    += p11.sum(0)                         # (b1,b2)
                    S_phibar_blk += phibar.sum(0)                      # (b1,b2)

                    # Empirical Y cross for this block (CPU)
                    YJ = Y[i2:i3, j0:j1]  # (B,b1) CPU
                    YK = Y[i2:i3, k0:k1]  # (B,b2) CPU
                    S_y11_blk += (YJ.T @ YK)                            # (b1,b2) CPU

                    # cleanup
                    del a, b, A, B, p11, phibar, quad, inv_const, one_m_r2, YJ, YK
                    if use_cuda:
                        torch.cuda.empty_cache()

                # ---- build model/empirical cov for this block ----
                C_model_blk = (S_p11_blk / n) - torch.outer(mp1, mp2)  # (b1,b2)
                C_emp_blk_cpu = (S_y11_blk / n) - torch.outer(
                    Ybar[j0:j1], Ybar[k0:k1]
                )  # CPU
                C_emp_blk = C_emp_blk_cpu.to(device=device, dtype=torch.float32, non_blocking=True)

                diff_blk   = C_model_blk - C_emp_blk                   # (b1,b2)
                phibar_blk = (S_phibar_blk / n)                        # (b1,b2)

                if j0 == k0:
                    # zero diagonal entries (not used in loss/grad)
                    diag_mask = torch.eye(b1, dtype=torch.bool, device=device)
                    diff_blk.masked_fill_(diag_mask, 0)
                    phibar_blk.masked_fill_(diag_mask, 0)
                    loss_val = loss_val + (torch.triu(diff_blk, diagonal=1)**2).sum()
                else:
                    loss_val = loss_val + (diff_blk**2).sum()

                # G = dL/d rho = 2 * diff * phibar  (symmetric)
                G_blk = 2.0 * diff_blk * phibar_blk                     # (b1,b2)

                # Accumulate gradient wrt V:
                # dL/d v1_rows += G_blk @ v2
                # dL/d v2_rows += G_blk.T @ v1
                Z[j0:j1] += G_blk @ v2                                  # (b1,k)
                Z[k0:k1] += G_blk.T @ v1                                # (b2,k)

                # cleanup block temporaries
                del S_p11_blk, S_phibar_blk, S_y11_blk, C_model_blk, C_emp_blk, C_emp_blk_cpu
                del diff_blk, phibar_blk, G_blk, rho_blk
                if use_cuda:
                    torch.cuda.empty_cache()

        # ---------------- Gradients & Adam updates ----------------------------
        grad_V = Z  # already sums both sides; corresponds to Σ_k G_{·k} v_k

        with torch.no_grad():
            # Adam for V
            m_V.mul_(beta1).add_(grad_V, alpha=1-beta1)
            v_V.mul_(beta2).addcmul_(grad_V, grad_V, value=1-beta2)
            lr_t = lr * math.sqrt(1 - beta2**(it+1)) / (1 - beta1**(it+1))
            V.addcdiv_(m_V, v_V.sqrt().add(epsilon), value=-lr_t)

            # ---- Project each row to ||v_j|| <= 1 (unit variance + PSD) ----
            norms = V.norm(dim=1, keepdim=True).clamp_min(1.0)
            V = V / norms  # rows with norm<1 unchanged; >=1 get scaled to 1

        loss_item = float(loss_val.detach().cpu())
        history['loss'].append(loss_item)
        with torch.no_grad():
            history['v_norm_max'].append(float(V.norm(dim=1).max().item()))

        if true_v is not None:
            tv = torch.as_tensor(true_v, dtype=torch.float32, device=device)
            # Optional: user-supplied metrics comparing Gram matrices
            metrics = full_kernel_metrics_tiled(V, tv, model_block=model_block)
            history['rmse_K'].append(metrics['rmse_K'])
            history['corr_K'].append(metrics['corr_K'])

        if loss_item < best_loss - 1e-9:
            best_loss, no_improve = loss_item, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping @ iter {it} (no improvement in {patience})")
                break

    # Return Γ (on CPU) and a dummy tau=1.0 placeholder for API compatibility
    return V.detach().cpu().numpy(), history


def neg_probit1pl(Y, n, m, lr=1e-2, max_iter=1000, patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # move data to torch
    Y_t = torch.as_tensor(Y, dtype=torch.float32, device=device)  # shape (n,m)

    # create parameters d_i and q_j with gradients
    d = torch.zeros(n, device=device, requires_grad=True)
    q = torch.zeros(m, device=device, requires_grad=True)

    # optimizer
    opt = torch.optim.Adam([d, q], lr=lr)

    # pre-make a standard normal for the CDF
    std_norm = torch.distributions.Normal(0.0, 1.0)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(max_iter)):
        opt.zero_grad()

        # 1) linear predictor μ_{ij} = q_j - d_i
        #    shape (n,m) via broadcasting
        mu = q.unsqueeze(0) - d.unsqueeze(1)

        # 2) probit link P_{ij} = Φ(μ_{ij})
        P = std_norm.cdf(mu).clamp(1e-6, 1-1e-6)

        # 3) log-likelihood term
        ll = (Y_t * torch.log(P) + (1.0 - Y_t) * torch.log(1.0 - P)).sum()

        # 4) Gaussian prior on q for identifiability
        ll = ll - 0.5 * (q**2).sum()

        # negative log-posterior
        loss = -ll

        # backprop + step
        loss.backward()
        opt.step()

        # re-center q to mean zero each step
        with torch.no_grad():
            q -= q.mean()

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d}  loss = {loss.item():.3f}")

    # detach to numpy if you like
    d_hat = d.detach().cpu().numpy()
    q_hat = q.detach().cpu().numpy()
    # Clean up GPU memory
    del d, q, Y_t, opt, std_norm, loss  # delete all GPU tensors/objects
    torch.cuda.empty_cache()
    gc.collect()

    return d_hat, q_hat


def second_stage_dq(
    Y, X, w,
    lr=1e-2, max_iter=500, tol=1e-4, patience=10,
    device=None, initial_d=None, initial_q=None
):
    """
    Fit a correlated‐error probit 1PL via pairwise composite likelihood.
    We hold `w` fixed (so Σ fixed), and learn difficulties d∈R^n and
    qualities q∈R^m by maximizing
      L_pair = ∏_i ∏_{j<k} P(Y_{ij},Y_{ik} | q_j, q_k, d_i, Σ[w])
    with Adam.

    Inputs
    ------
    Y : array-like (n, m) of {0,1}
    X : array-like (m, p) feature matrix for models
    w : array-like (p,)   -- your learned kernel weights

    Returns
    -------
    d_hat : (n,) estimated difficulties
    q_hat : (m,) estimated qualities (mean-zero)
    history : dict with keys 'd', 'q', 'loss'
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # data tensors
    Y = torch.as_tensor(Y, dtype=torch.float32, device=device)
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    w = torch.as_tensor(w, dtype=torch.float32, device=device)

    n, m = Y.shape
    p = X.shape[1]

    # precompute the fixed correlation matrix ρ_{jk}
    diff3 = X[:, None, :] - X[None, :, :]             # (m, m, p)
    sq    = (diff3.pow(2) * w[None, None, :]).sum(2)  # (m, m)
    rho   = torch.exp(-sq)
    eps_rho = 1e-6
    rho = torch.clamp(rho, -1.0+eps_rho, 1.0-eps_rho)  # (m,m)

    # mask for upper‐triangle pairs j<k
    pair_mask = torch.triu(torch.ones(m, m, dtype=torch.bool, device=device), diagonal=1)

    # parameters to learn
    d = torch.zeros(n, device=device, requires_grad=True)
    q = torch.zeros(m, device=device, requires_grad=True)
    if initial_d is not None:  d.data[:] = torch.as_tensor(initial_d, device=device)
    if initial_q is not None:  q.data[:] = torch.as_tensor(initial_q, device=device)

    optimizer = torch.optim.Adam([d, q], lr=lr)

    history = {"d": [d.detach().cpu().numpy()],
               "q": [q.detach().cpu().numpy()],
               "loss": [float("inf")]}

    best_loss = float("inf")
    no_improve = 0

    for it in tqdm(range(1, max_iter+1)):
        optimizer.zero_grad()

        # 1) linear predictors and univariate CDF
        mu     = q.unsqueeze(0) - d.unsqueeze(1)     # (n, m)
        Phi_mu = torch.special.ndtr(mu)              # Φ(q_j - d_i)

        # 2) bivariate probabilities
        a  = mu.unsqueeze(2)  # (n, m, 1)
        b  = mu.unsqueeze(1)  # (n, 1, m)
        p11 = bi_cdf(a, b, rho)                       # (n, m, m)
        Pj  = Phi_mu.unsqueeze(2)                     # (n, m, 1)
        Pk  = Phi_mu.unsqueeze(1)                     # (n, 1, m)
        p10 = Pj - p11
        p01 = Pk - p11
        p00 = 1.0 - Pj - Pk + p11

        # 3) select the right cell for each observed pair
        Yj = Y.unsqueeze(2)  # (n, m, 1)
        Yk = Y.unsqueeze(1)  # (n, 1, m)
        P   = torch.where((Yj==1)&(Yk==1), p11,
              torch.where((Yj==1)&(Yk==0), p10,
              torch.where((Yj==0)&(Yk==1), p01, p00)))
        
        # 4) extract j<k, clamp, log, sum
        P_pairs = P[:, pair_mask]                    # (n, n_pairs)
        eps = torch.finfo(P_pairs.dtype).eps
        P_pairs = torch.clamp(P_pairs, eps, 1.0)
        loss = - torch.log(P_pairs).sum()             # negative composite log-lik

        # backward + step
        loss.backward()
        optimizer.step()
        # re-center q for identifiability
        with torch.no_grad():
            q -= q.mean()

        loss_val = loss.item()
        history["loss"].append(loss_val)
        history["d"].append(d.detach().cpu().numpy())
        history["q"].append(q.detach().cpu().numpy())

        # early-stop on no improvement
        if loss_val + tol < best_loss:
            best_loss = loss_val
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[stop at iter {it}] no loss improvement in {patience} iters.")
                break

    d_hat = d.detach().cpu().numpy()
    q_hat = q.detach().cpu().numpy()
    return d_hat, q_hat, history

def neg_probit1pl_batched(Y, n, m,
                         lr=1e-2, max_iter=1000, patience=40,
                         batch_size=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) send Y to GPU, build mask for non-nan, replace nan→0
    Y_t = torch.as_tensor(Y, dtype=torch.float32, device=device)  # (n,m)
    mask = ~torch.isnan(Y_t)
    Y_t  = torch.nan_to_num(Y_t, nan=0.0)

    # 2) parameters
    d = torch.zeros(n, requires_grad=True, device=device)
    q = torch.zeros(m, requires_grad=True, device=device)

    opt      = torch.optim.Adam([d, q], lr=lr)
    std_norm = torch.distributions.Normal(0.0, 1.0)

    best_loss, patience_counter = float('inf'), 0

    for epoch in tqdm(range(max_iter), desc="probit1PL"):
        opt.zero_grad()
        loss_total = 0.0

        # 3) batch over items
        for i0 in range(0, n, batch_size):
            i1 = min(n, i0 + batch_size)

            # a) compute μ and P only for this chunk
            mu    = q.unsqueeze(0) - d[i0:i1].unsqueeze(1)      # (B, m)
            P     = std_norm.cdf(mu).clamp(1e-6, 1 - 1e-6)      # (B, m)

            # b) compute chunk log‐likelihood (only observed entries)
            Yc    = Y_t[i0:i1]                                  # (B, m)
            Mc    = mask[i0:i1]                                # (B, m)
            llc   = (Yc * torch.log(P) + (1 - Yc) * torch.log(1 - P))
            llc   = (Mc * llc).sum()                           # scalar

            # c) negative chunk loss & backward
            chunk_loss = -llc
            chunk_loss.backward()                              # accumulates grads
            loss_total += chunk_loss.item()

            # free that chunk’s graph
            del mu, P, Yc, Mc, llc, chunk_loss
            torch.cuda.empty_cache()

        # 4) add prior on q:  ½·∥q∥²  → grad += q
        prior_term = 0.5 * (q**2).sum().item()
        loss_total += prior_term
        q.grad = q.grad + q  # add derivative of prior

        # 5) step & recenter
        opt.step()
        with torch.no_grad():
            q -= q.mean()

        # 6) early‐stop
        if loss_total < best_loss - 1e-6:
            best_loss, patience_counter = loss_total, 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d}  loss={loss_total:.3f}")

    # detach and cleanup
    d_hat = d.detach().cpu().numpy()
    q_hat = q.detach().cpu().numpy()

    del d, q, Y_t, mask, opt, std_norm
    torch.cuda.empty_cache()
    gc.collect()

    return d_hat, q_hat

# Y = pd.read_csv('data/helm/helm_ans.csv')

def main(data_path, fp_out):
    Y = pd.read_csv(data_path)
    Y = Y.drop(columns=['question_id']).to_numpy().astype(float)
    n = Y.shape[0]
    m = Y.shape[1]
    d_hat, q_hat = neg_probit1pl_batched(Y, n, m, lr=lr)
    res = {'d': d_hat, 'q': q_hat}
    with open(fp_out, 'wb') as f:
        pickle.dump(res, f)


if __name__ == "__main__":
    main(data_path='data/helm/helm_ans.csv', fp_out='results/helm/helm_latent_p10.pkl')
    main(data_path='data/helm/helm_mult_choice_ans.csv', fp_out='results/helm/helm_mult_choice.pkl')

    main(data_path='data/hf/hf_ans.csv', fp_out='results/hf/hf_k1_p15.pkl')
    main(data_path='data/hf/hf_mult_choice_ans.csv', fp_out='results/hf/hf_mult_choice.pkl')