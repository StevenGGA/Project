import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingRegressor

# LOAD DATA & DEFINE FEATURES & GROUPS
df = pd.read_csv('Spotify_Youtube.csv')

numeric_features = [
    'Danceability','Energy','Key','Loudness','Speechiness',
    'Acousticness','Instrumentalness','Liveness','Valence','Tempo',
    'Duration_ms', 'Likes', 'Views', 'Comments'
]
target_reg = 'Views'  # or 'Comments' or 'Likes'
numeric_features.remove(target_reg)
groups_full = df['Album']

# PREPARE REGRESSION DATA
X = df[numeric_features].copy()
y = np.log1p(df[target_reg]) # log1p transform target
mask = y.notna() # drop missing targets
X, y = X[mask], y[mask]
groups = groups_full[mask]

# HOLD-OUT SPLIT (80/20) WITH GROUPS
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]

# PIPELINE & GRID-SEARCH OPTIMIZING RMSLE
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',   StandardScaler()),
    ('model',    HistGradientBoostingRegressor(random_state=42))
])

param_grid = {
    'model__max_iter':          [100, 200, 400],
    'model__learning_rate':     [0.01, 0.1, 0.2],
    'model__max_depth':         [3, None],
    'model__l2_regularization': [0.0, 0.1, 1.0, 2.0],
}

inner_cv = GroupKFold(n_splits=3)
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=inner_cv,
    scoring='neg_mean_squared_log_error',
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train, groups=groups_train)

# EVALUATE ON HOLD-OUT
y_pred_log = grid.predict(X_test)
holdout_rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred_log))
y_pred_raw = np.expm1(y_pred_log)
holdout_rmse = np.sqrt(mean_squared_error(np.expm1(y_test), y_pred_raw))

print("Best params:       ", grid.best_params_)
print(f"Hold-out RMSLE:    {holdout_rmsle:.4f}")
print(f"Hold-out RMSE:     {holdout_rmse:.2f}")

# PERMUTATION IMPORTANCE
best_pipe = grid.best_estimator_
perm = permutation_importance(
    best_pipe,
    X_test,
    y_test,
    scoring='neg_mean_squared_log_error',
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

imp_df = (
    pd.DataFrame({
        'feature':    numeric_features,
        'importance_mean': perm.importances_mean,
        'importance_std': perm.importances_std
    })
    .sort_values('importance_mean', ascending=False)
    .reset_index(drop=True)
)

print("\nPermutation Importances:")
print(imp_df)
