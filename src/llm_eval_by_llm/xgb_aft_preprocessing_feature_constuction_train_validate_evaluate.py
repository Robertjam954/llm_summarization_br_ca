import pandas as pd
import numpy as np
import xgboost as xgb
from lifelines.utils import concordance_index

# --- Data Preprocessing ---
# Load the dataset (assumed to be pre-cleaned and merged with necessary covariates)
df = pd.read_csv('path/to/cleaned_data.csv')  # Update with actual data path

# Remove any TBL-related features if present
cols_to_drop = [col for col in df.columns if col.upper().startswith('TBL_')]
if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"Dropped TBL-related features: {cols_to_drop}")

# Remove or ignore quartile/binned features for specific variables 
quartile_cols = ['aneuploidy_score_quartile', 'buffa_hypoxia_quartile', 
                 'variant_allele_count_quartile', 'mantis_bin']
for col in quartile_cols:
    if col in df.columns:
        df.drop(columns=col, inplace=True)
        print(f"Dropped binned/quartile feature: {col}")

# Ensure certain important features are treated as continuous numeric
continuous_cols = ['aneuploidy_score', 'mantis', 'buffa_hypoxia_score', 'variant_allele_count']
for col in continuous_cols:
    if col in df.columns:
        # Convert to numeric (this will turn non-numeric entries to NaN, if any)
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle missing values in critical fields (e.g., drop rows with missing survival time or event)
# Assume 'survival_time' is the time-to-event (or censoring) and 'event' is the event indicator (1=event occurred, 0=censored)
df = df.dropna(subset=['survival_time', 'event'])
print(f"Data shape after dropping missing target values: {df.shape}")

# --- Feature Construction ---
# One-hot encode categorical features like subtype and AJCC stage (if they exist)
if 'subtype' in df.columns:
    subtype_dummies = pd.get_dummies(df['subtype'], prefix='subtype')
    df = pd.concat([df, subtype_dummies], axis=1)
    print("One-hot encoded subtype categories:", list(subtype_dummies.columns))
if 'ajcc_pathologic_tumor_stage' in df.columns:
    stage_dummies = pd.get_dummies(df['ajcc_pathologic_tumor_stage'], prefix='stage')
    df = pd.concat([df, stage_dummies], axis=1)
    print("One-hot encoded AJCC stage categories:", list(stage_dummies.columns))

# Identify feature columns for modeling.
# We'll include continuous covariates and relevant binary indicators (top genes, etc.)
feature_cols = []

# Add continuous numeric covariates if present
for col in continuous_cols:
    if col in df.columns:
        feature_cols.append(col)

# Add top 10 gene indicator features (columns starting with "G__")
gene_indicator_cols = [col for col in df.columns if col.startswith('G__')]
feature_cols.extend(gene_indicator_cols)

# Add subtype and stage one-hot columns (if they were created)
if 'subtype' in df.columns:
    feature_cols.extend([col for col in df.columns if col.startswith('subtype_')])
if 'ajcc_pathologic_tumor_stage' in df.columns:
    feature_cols.extend([col for col in df.columns if col.startswith('stage_')])

print(f"Total features used for modeling: {len(feature_cols)}")

# Define X matrix and survival labels (time and event)
X = df[feature_cols].values
times = df['survival_time'].values  # time-to-event or time-to-censor
events = df['event'].astype(int).values  # event indicator (1 if event occurred, 0 if censored)

# Convert survival times to the (lower_bound, upper_bound) format required by XGBoost AFT
# Uncensored events have lower_bound == upper_bound == event time
# Right-censored events have lower_bound = censor time, upper_bound = +inf
y_lower = times.copy()
y_upper = times.copy()
y_upper[events == 0] = np.inf  # for censored cases, set upper bound to infinity

# --- Training/Validation Split ---
from sklearn.model_selection import train_test_split
X_train, X_val, y_lower_train, y_lower_val, y_upper_train, y_upper_val, events_train, events_val = train_test_split(
    X, y_lower, y_upper, events, test_size=0.2, random_state=42
)
print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

# --- Model Setup and Hyperparameter Tuning ---
# Prepare DMatrix for XGBoost (AFT requires using xgboost DMatrix API to set labels)
dtrain = xgb.DMatrix(X_train)
dtrain.set_float_info('label_lower_bound', y_lower_train)
dtrain.set_float_info('label_upper_bound', y_upper_train)
dval = xgb.DMatrix(X_val)
dval.set_float_info('label_lower_bound', y_lower_val)
dval.set_float_info('label_upper_bound', y_upper_val)

# Set initial XGBoost parameters for the AFT survival model
params = {
    'objective': 'survival:aft',
    'eval_metric': 'aft-nloglik',            # negative log-likelihood for AFT:contentReference[oaicite:0]{index=0}
    'aft_loss_distribution': 'normal',       # distribution for the AFT noise term
    'aft_loss_distribution_scale': 1.0,      # scaling factor for the distribution
    'tree_method': 'hist',                   # use efficient histogram tree method
    'learning_rate': 0.05,
    'max_depth': 6,
    'seed': 42                               # for reproducibility
}

# Use cross-validation to find optimal number of boosting rounds (trees) with early stopping
print("Starting cross-validation for optimal boosting rounds...")
cv_results = xgb.cv(
    params, dtrain,
    num_boost_round=1000,
    nfold=5,
    early_stopping_rounds=10,
    metrics="aft-nloglik",
    seed=42
)
best_rounds = len(cv_results)  # number of boosting rounds before early stopping
print(f"Optimal number of boosting rounds (trees) from CV: {best_rounds}")

# (Optional) Hyperparameter tuning:
# You could perform a random search over params here by iterating over a set of parameter combinations
# and using xgb.cv to evaluate performance. This is omitted for brevity. 
# Example (conceptual):
# for param_combo in random_param_grid:
#     cv_result = xgb.cv(param_combo, dtrain, num_boost_round=500, nfold=5, early_stopping_rounds=10, metrics="aft-nloglik")
#     ... # select combo with lowest aft-nloglik

# --- Model Training ---
# Train the final model using the best number of rounds, with early stopping on the validation set
print("Training final model...")
bst = xgb.train(
    params, 
    dtrain, 
    num_boost_round=best_rounds, 
    evals=[(dval, "validation")],
    early_stopping_rounds=10,
    verbose_eval=50
)

# --- Model Evaluation ---
# Predict risk/survival scores on the validation set
# (For AFT, predictions are typically the predicted log survival time or similar)
y_pred_val = bst.predict(dval)

# If using a log-linear AFT model (e.g., normal distribution), we can exponentiate predictions to get times
predicted_time_val = np.exp(y_pred_val)  # this is an estimate of median survival time for each instance

# Calculate Concordance Index (C-index) on validation set
# We provide true times, predicted scores, and event indicators to the concordance_index function
c_index_val = concordance_index(
    event_times=y_lower_val,             # true event/censoring times
    predicted_scores= -y_pred_val,       # use negative of predicted value so that higher risk = lower predicted time
    event_observed=events_val
)
# (We negate y_pred_val because a higher predicted value (log-time) means longer survival, i.e. lower risk. 
# Using -y_pred makes it so that a higher score corresponds to higher risk of earlier event, for concordance calculation.)

# Calculate RMSE on the validation set for uncensored cases (where event occurred)
from sklearn.metrics import mean_squared_error
# Only consider cases where event_observed == 1 for a fair RMSE (actual event times known)
rmse_val = np.sqrt(mean_squared_error(y_lower_val[events_val == 1], predicted_time_val[events_val == 1]))

print(f"Validation C-index: {c_index_val:.3f}")
print(f"Validation RMSE (for uncensored events): {rmse_val:.3f}")

# --- Artifact Saving ---
# Save the trained model to a file for future use
model_filename = "xgb_aft_survival_model.json"
bst.save_model(model_filename)
print(f"Model saved to {model_filename}")
