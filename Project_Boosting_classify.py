import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingClassifier
# LOAD DATA & DEFINE FEATURES & GROUPS
df = pd.read_csv('Spotify_Youtube.csv')
numeric_features = [
    'Danceability','Energy','Key','Loudness','Speechiness',
    'Acousticness','Instrumentalness','Liveness','Valence','Tempo',
    'Duration_ms', 'Likes', 'Views', 'Comments'
]
groups_full = df['Album']  # grouping for split


# CLASSIFICATION: HOLD-OUT SPLIT + NESTED CV
# Prepare X, y
X_clf = df[numeric_features]
y_clf = (df['Album_type'] == 'single').astype(int)
mask_clf = df['Album_type'].notna()
X_clf = X_clf[mask_clf]
y_clf = y_clf[mask_clf]
groups_clf = groups_full[mask_clf]

# Single grouped train/test split
gss_clf = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss_clf.split(X_clf, y_clf, groups_clf))
X_clf_train, X_clf_test = X_clf.iloc[train_idx], X_clf.iloc[test_idx]
y_clf_train, y_clf_test = y_clf.iloc[train_idx], y_clf.iloc[test_idx]
groups_clf_train = groups_clf.iloc[train_idx]

# PIPELINE & GRID-SEARCH OPTIMIZING RMSLE
pipe_clf = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler',   StandardScaler()),
    ('model',    HistGradientBoostingClassifier(random_state=42))
])

param_grid_clf = {
    'model__max_iter':          [100, 200, 400],
    'model__learning_rate':     [0.01, 0.1, 0.2],
    'model__max_depth':         [3, None],
    'model__l2_regularization': [0.0, 0.1, 1.0],
}

inner_cv = GroupKFold(n_splits=3)
grid_clf = GridSearchCV(
    estimator=pipe_clf,
    param_grid=param_grid_clf,
    cv=inner_cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_clf.fit(X_clf_train, y_clf_train, groups=groups_clf_train)
y_clf_pred = grid_clf.predict(X_clf_test)
acc = accuracy_score(y_clf_test, y_clf_pred)
print("Final hold-out Accuracy (single vs album):", acc)

# PERMUTATION IMPORTANCE
result = permutation_importance(
    grid_clf, X_clf_test, y_clf_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

# Build a sorted DataFrame
imp_df = pd.DataFrame({
    'feature': numeric_features,
    'importance_mean': result.importances_mean,
    'importance_std': result.importances_std
}).sort_values('importance_mean', ascending=False).reset_index(drop=True)

print("\nPermutation Importances:")
print(imp_df)