from infer_q_d import neg_probit1pl_batched
import pandas as pd
import pickle
import numpy as np
from scipy.stats import norm
import argparse

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
args = parser.parse_args()

max_iter = args.max_iter
lr = args.lr
R = args.R
n = args.n
m = args.m
num_covariates = args.num_covariates
w = args.w
subsample_frac = args.subsample_frac
k = args.k


### MULTIPLICITY experiments
base = 'CA_2018_all_predictions_test'

results = {}
df = pd.read_csv(f'results/relativity_population/{base}.csv')
dt_cols = [i for i in df.columns if 'randomforestclassifier' in i]
logistic_cols = [i for i in df.columns if 'logisticregression' in i]
mlp_cols = [i for i in df.columns if 'mlpclassifier' in i]


Y = pd.concat([df[dt_cols[:10]], df[dt_cols[20:30]]], axis=1).to_numpy().astype(float)
n = Y.shape[0]
m = Y.shape[1]
# print(n, m)
save_path = f'results/relativity_population/acsincome_only_rf_highly_correlated.pkl'

d_hat, q_hat = neg_probit1pl_batched(Y, n, m, lr=lr)
results['d'] = d_hat
results['q'] = q_hat
with open(f'{save_path}', 'wb') as f:
    pickle.dump(results, f)
v_hat, history = fit_kernel_latent_stream(Y, np.random.standard_normal(size=(m, num_covariates)), q_hat, d_hat, lr=lr)
results['v'] = v_hat
with open(f'{save_path}', 'wb') as f:
    pickle.dump(results, f)

Y = pd.concat([df[dt_cols[:10]], df[dt_cols[20:30]], df[logistic_cols]], axis=1).to_numpy().astype(float)
n = Y.shape[0]
m = Y.shape[1]
# print(n, m)
save_path = f'results/relativity_population/acsincome_rf_highly_correlated_logistic_small.pkl'

d_hat, q_hat = neg_probit1pl_batched(Y, n, m, lr=lr)
results['d'] = d_hat
results['q'] = q_hat
with open(f'{save_path}', 'wb') as f:
    pickle.dump(results, f)
v_hat, history = fit_kernel_latent_stream(Y, np.random.standard_normal(size=(m, num_covariates)), q_hat, d_hat, lr=lr)
results['v'] = v_hat
with open(f'{save_path}', 'wb') as f:
    pickle.dump(results, f)



Y = pd.concat([df[dt_cols[:10]], df[dt_cols[20:30]], df[logistic_cols], df[mlp_cols]], axis=1).to_numpy().astype(float)
n = Y.shape[0]
m = Y.shape[1]
save_path = f'results/relativity_population/acsincome_rf_highly_correlated_logistic_small_mlp.pkl'

d_hat, q_hat = neg_probit1pl_batched(Y, n, m, lr=lr)
results['d'] = d_hat
results['q'] = q_hat
with open(f'{save_path}', 'wb') as f:
    pickle.dump(results, f)
v_hat, history = fit_kernel_latent_stream(Y, np.random.standard_normal(size=(m, num_covariates)), q_hat, d_hat, lr=lr)
results['v'] = v_hat
with open(f'{save_path}', 'wb') as f:
    pickle.dump(results, f)

Y = pd.read_csv('results/relativity_population/acsincome_rf_highly_correlated_logistic_small_mlp_d0.csv')
Y = Y.to_numpy().astype(float)
n = Y.shape[0]
m = Y.shape[1]
print(n, m)
save_path = f'results/relativity_population/acsincome_rf_highly_correlated_logistic_small_mlp_d0.pkl'

d_hat, q_hat = neg_probit1pl_batched(Y, n, m, lr=lr)
results['d'] = d_hat
results['q'] = q_hat
with open(f'{save_path}', 'wb') as f:
    pickle.dump(results, f)




# HELM experiments
starting_models = ['openai/gpt-4-0613',
       'openai/gpt-4-1106-preview', 'openai/gpt-4-turbo-2024-04-09',
       'openai/gpt-4o-2024-05-13', 'openai/gpt-4o-2024-08-06']
next_correlated_models = ['anthropic/claude-3-5-sonnet-20240620',
       'anthropic/claude-3-5-sonnet-20241022',
       'anthropic/claude-3-haiku-20240307',
       'anthropic/claude-3-opus-20240229', 'writer/palmyra-x-004', 'writer/palmyra-x-v3']
neg_correlated_models = ['meta/llama-3.1-8b-instruct-turbo', 'meta/llama-3.2-11b-vision-instruct-turbo', 'allenai/olmo-1.7-7b', 'allenai/olmo-7b',
'cohere/command-r', 'cohere/command-r-plus']

df = pd.read_csv('data/helm/helm_ans.csv')
Y = df[starting_models].to_numpy().astype(float)
n = Y.shape[0]
m = Y.shape[1]

fp_out = f'results/multiplicity/helm_exp1.pkl'
d_hat, q_hat = neg_probit1pl_batched(Y, n, m, lr=lr)
results['d'] = d_hat
results['q'] = q_hat
with open(fp_out, 'wb') as f:
    pickle.dump(results, f)

v_hat, history = fit_kernel_latent_stream(Y, np.random.standard_normal(size=(m, num_covariates)), q_hat, d_hat, lr=lr)
results['v'] = v_hat
with open(fp_out, 'wb') as f:
    pickle.dump(results, f)


fp_out = f'results/multiplicity/helm_exp2.pkl'
Y = df[starting_models + next_correlated_models].to_numpy().astype(float)
n = Y.shape[0]
m = Y.shape[1]

d_hat, q_hat = neg_probit1pl_batched(Y, n, m, lr=lr)
results['d'] = d_hat
results['q'] = q_hat
with open(fp_out, 'wb') as f:
    pickle.dump(results, f)

v_hat, history = fit_kernel_latent_stream(Y, np.random.standard_normal(size=(m, num_covariates)), q_hat, d_hat, lr=lr)
results['v'] = v_hat
with open(fp_out, 'wb') as f:
    pickle.dump(results, f)


fp_out = f'results/multiplicity/helm_exp3.pkl'
Y = df[starting_models + next_correlated_models + neg_correlated_models].to_numpy().astype(float)
n = Y.shape[0]
m = Y.shape[1]

d_hat, q_hat = neg_probit1pl_batched(Y, n, m, lr=lr)
results['d'] = d_hat
results['q'] = q_hat
with open(fp_out, 'wb') as f:
    pickle.dump(results, f)

v_hat, history = fit_kernel_latent_stream(Y, np.random.standard_normal(size=(m, num_covariates)), q_hat, d_hat, lr=lr)
results['v'] = v_hat
with open(fp_out, 'wb') as f:
    pickle.dump(results, f)