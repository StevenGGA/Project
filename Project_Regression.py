import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error

# LOAD DATA & DEFINE FEATURES & GROUPS
df = pd.read_csv('Spotify_Youtube.csv')
numeric_features = [
    'Danceability','Energy','Key','Loudness','Speechiness',
    'Acousticness','Instrumentalness','Liveness','Valence','Tempo',
    'Duration_ms', 'Likes', 'Views', 'Comments'
]
target_reg = 'Likes' # Change to Likes or Comments for different target
numeric_features.remove(target_reg)
groups_full = df['Album']

# PREPARE REGRESSION DATA
X = df[numeric_features]
y = np.log1p(df[target_reg]) # log1p transform target
mask = y.notna() # drop missing targets
X = X[mask]
y = y[mask]
groups = groups_full[mask]

# HOLD-OUT SPLIT (80/20) WITH GROUPING
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]

# PIPELINE + GRID SEARCH OPTIMIZING RMSLE
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',   StandardScaler()),
    ('model',    RandomForestRegressor(random_state=42))
])

param_grid = {
    'model__n_estimators': [100, 300],
    'model__max_depth':    [None, 10, 30],
    'model__max_features': ['sqrt', 'log2']
}

inner_cv = GroupKFold(n_splits=3)
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=inner_cv,
    scoring='neg_mean_squared_log_error',  # optimize for RMSLE
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train, groups=groups_train)

# EVALUATE ON HOLD-OUT
y_pred_log = grid.predict(X_test)
holdout_rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred_log))
y_pred_raw = np.expm1(y_pred_log)
holdout_rmse = np.sqrt(mean_squared_error(np.expm1(y_test), y_pred_raw))

print(f"Best params: {grid.best_params_}")
print(f"Hold-out RMSLE: {holdout_rmsle:.4f}")
print(f"Hold-out RMSE (raw views): {holdout_rmse:.2f}")


# FEATURE IMPORTANCES
# Regression importances
best_reg = grid.best_estimator_.named_steps['model']
imp_reg = best_reg.feature_importances_
imp_reg_df = pd.DataFrame({'feature': numeric_features, 'importance': imp_reg})
imp_reg_df = imp_reg_df.sort_values('importance', ascending=False).reset_index(drop=True)
print("\nRegression feature importances:")
print(imp_reg_df)