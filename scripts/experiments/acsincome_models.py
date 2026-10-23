# from folktables import ACSDataSource, ACSEmployment, ACSIncome
import pandas as pd
import numpy as np
import scipy
import os
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def get_param_string(params):
    """
    Convert model parameters to a condensed string representation.
    Excludes random_state and common default parameters.
    """
    # Skip random_state and parameters with common default values
    skip_params = {'random_state', 'verbose', 'n_jobs'}
    param_strs = []
    
    for k, v in sorted(params.items()):
        if k not in skip_params:
            # Shorten parameter names
            k_short = k.replace('max_', 'm').replace('min_', 'n').replace('n_estimators', 'ne')
            # Format value, handling different types
            if isinstance(v, float):
                v_str = f"{v:.1e}" if v < 0.01 else f"{v:.1f}"
            else:
                v_str = str(v)
            param_strs.append(f"{k_short}{v_str}")
    
    return "_".join(param_strs) if param_strs else "default"

def train_model(model_class, num_models, rng, model_params=None):
    """
    Train multiple instances of a model and collect their predictions.
    
    Args:
        model_class: The class of the model (e.g., RandomForestClassifier, LogisticRegression)
        num_models: Number of model instances to train
        rng: Random number generator for reproducibility
        model_params: Dictionary of parameters to pass to the model constructor
    """
    if model_params is None:
        model_params = {}

    # if model_class is not KNeighborsClassifier:
    #     # Always ensure random_state is set for reproducibility
    #     model_params['random_state'] = rng
    
    all_predictions = pd.DataFrame()
    base_name = model_class.__name__.lower()  # Get name from class
    param_str = get_param_string(model_params)
    model_name = f"{base_name}_{param_str}"
    
    for i in tqdm(range(num_models)):
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(features_test)
        all_predictions[f'{model_name}_{i+1}'] = y_pred == label_test

    return all_predictions

year = '2018'

# training
state = 'CA'

df = pd.read_csv(f'data/acsincome/{state}_{year}_1yr.csv')
label = df['label'].to_numpy()
group = df['group'].to_numpy()
features = df.drop(columns=['label', 'group']).to_numpy()

X_train, features_test, y_train, label_test = train_test_split(
    features, label, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
features_test = scaler.transform(features_test)

rng = np.random.RandomState(42)
all_predictions = pd.DataFrame()

save_path = f'results/relativity_population/{state}_{year}_all_predictions_test.csv'


num_models = 5

# # RandomForest with different parameters
# print("Training Random Forest models...")
list_of_n_estimators = [50, 100]
list_of_max_depth = [5, 10]
list_of_criterion = ['gini', 'entropy']
for n_estimators in list_of_n_estimators:
    for max_depth in list_of_max_depth:
        for criterion in list_of_criterion:
            rf_params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'criterion': criterion}
            buffer = train_model(RandomForestClassifier, num_models, rng, rf_params)
            all_predictions = pd.concat([all_predictions, buffer], axis=1)

all_predictions.to_csv(save_path, index=False)

rng = np.random.RandomState(42)

num_models = 1

print("Training Logistic Regression models...")
# LogisticRegression with different regularization strengths
lr_params1 = {'max_iter': 1000, 'C': 0.1}
lr_params2 = {'max_iter': 1000, 'C': 1.0}
buffer = train_model(LogisticRegression, num_models, rng, lr_params1)
all_predictions = pd.concat([all_predictions, buffer], axis=1)

buffer = train_model(LogisticRegression, num_models, rng, lr_params2)
all_predictions = pd.concat([all_predictions, buffer], axis=1)

num_models = 5
rng = np.random.RandomState(42)
print("Training MLPClassifier models...")
# MLPClassifier
mlp_params1 = {'hidden_layer_sizes': (100,), 'max_iter': 10000}
mlp_params2 = {'hidden_layer_sizes': (20, 10), 'max_iter': 10000}
mlp_params3 = {'hidden_layer_sizes': (10, 5, 2), 'max_iter': 10000}
buffer = train_model(MLPClassifier, num_models, rng, mlp_params1)
all_predictions = pd.concat([all_predictions, buffer], axis=1)

buffer = train_model(MLPClassifier, num_models, rng, mlp_params2)
all_predictions = pd.concat([all_predictions, buffer], axis=1)

buffer = train_model(MLPClassifier, num_models, rng, mlp_params3)
all_predictions = pd.concat([all_predictions, buffer], axis=1)

all_predictions.to_csv(save_path, index=False)

# "+ d0" population for Fig. 3: the 20 highly correlated RFs, LR, and MLPs above,
# plus 25 artificial models that each succeed independently on every test example
# with a model-specific probability p ~ N(0.7, 0.2) (clipped to [0, 1]).
rf_cols = [i for i in all_predictions.columns if 'randomforestclassifier' in i]
logistic_cols = [i for i in all_predictions.columns if 'logisticregression' in i]
mlp_cols = [i for i in all_predictions.columns if 'mlpclassifier' in i]
d0_predictions = pd.concat([all_predictions[rf_cols[:10]], all_predictions[rf_cols[20:30]],
                            all_predictions[logistic_cols], all_predictions[mlp_cols]], axis=1)

num_d0_models = 25
rng = np.random.RandomState(42)
p_d0 = np.clip(rng.normal(loc=0.7, scale=0.2, size=num_d0_models), 0, 1)
n_test = len(label_test)
for j, p in enumerate(p_d0):
    d0_predictions[f'q{p:.2f}_{j+1}'] = (rng.uniform(size=n_test) < p).astype(int)

d0_predictions.to_csv('results/relativity_population/acsincome_rf_highly_correlated_logistic_small_mlp_d0.csv', index=False)
