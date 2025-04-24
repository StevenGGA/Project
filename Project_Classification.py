import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, mean_squared_log_error

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

# Pipeline and hyperparameter grid
pipe_clf = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler',   StandardScaler()),
    ('model',    RandomForestClassifier(random_state=42))
])
param_grid_clf = {
    'model__n_estimators': [100, 300], #Try 500, 1000 for more trees
    'model__max_depth':    [None, 10, 30], #Try 50, 100 for larger trees
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

# Fit on train, evaluate on hold-out
grid_clf.fit(X_clf_train, y_clf_train, groups=groups_clf_train)
y_clf_pred = grid_clf.predict(X_clf_test)
acc = accuracy_score(y_clf_test, y_clf_pred)
print("Final hold-out Accuracy (single vs album):", acc)



# Classification importances
best_clf = grid_clf.best_estimator_.named_steps['model']
imp_clf = best_clf.feature_importances_
imp_clf_df = pd.DataFrame({'feature': numeric_features, 'importance': imp_clf})
imp_clf_df = imp_clf_df.sort_values('importance', ascending=False).reset_index(drop=True)
print("\nClassification feature importances:")
print(imp_clf_df)
