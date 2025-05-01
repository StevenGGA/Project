import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


# LOAD DATA & DEFINE BASE FEATURES
df = pd.read_csv('Spotify_Youtube.csv')
numeric_base = [
    'Danceability','Energy','Key','Loudness','Speechiness',
    'Acousticness','Instrumentalness','Liveness','Valence','Tempo',
    'Duration_ms','Likes','Views','Comments'
]
groups = df['Album']
y = (df['Album_type'] == 'single').astype(int)
mask = df['Album_type'].notna()


# FEATURE ENGINEERING METHODS
# Original features
orig_feats = numeric_base.copy()

# Ratios + interactions + logs
df['like_ratio'] = df['Likes'] / df['Views'].replace(0, np.nan)
df['comment_ratio'] = df['Comments'] / df['Views'].replace(0, np.nan)
df['energy_loudness'] = df['Energy'] * df['Loudness']
df['dance_valence'] = df['Danceability'] * df['Valence']
df['log_views'] = np.log1p(df['Views'])
df['log_likes'] = np.log1p(df['Likes'])
eng_feats = orig_feats + [
    'like_ratio','comment_ratio',
    'energy_loudness','dance_valence',
    'log_views','log_likes'
]

# PCA on audio+engagement (with imputation)
median_vals = df[numeric_base].median()
filled = df.loc[mask, numeric_base].fillna(median_vals)  # impute missing before PCA
scaler = StandardScaler()
audio_eng = scaler.fit_transform(filled)
pca = PCA(n_components=5, random_state=42)
pcs = pca.fit_transform(audio_eng)
for i in range(5):
    df.loc[mask, f'pc{i+1}'] = pcs[:, i]
pca_feats = [f'pc{i+1}' for i in range(5)]

# Combined engineered + PCA
combo_feats = eng_feats + pca_feats

# Smart feature
df['log_duration'] = np.log1p(df['Duration_ms'])
df['loud_energy'] = df['Loudness'] * df['Energy']
df['valence_acoustic'] = df['Valence'] * df['Acousticness']
median_views = df['Views'].median()
df['high_view_flag'] = (df['Views'] > median_views).astype(int)
smart_feats = numeric_base + [
    'log_duration','loud_energy','valence_acoustic','high_view_flag'
]

#EVALUATION FUNCTION
def evaluate(features):
    X = df.loc[mask, features]
    y_sub = y[mask]
    groups_sub = groups[mask]
    # hold-out split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y_sub, groups_sub))
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y_sub.iloc[tr_idx], y_sub.iloc[te_idx]
    grp_tr = groups_sub.iloc[tr_idx]
    # pipeline + grid search
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',   StandardScaler()),
        ('model',    HistGradientBoostingClassifier(random_state=42))
    ])
    grid = GridSearchCV(
        estimator=pipe,
        param_grid={
            'model__max_iter':          [100, 200, 400],
            'model__learning_rate':     [0.01, 0.1, 0.2],
            'model__max_depth':         [None],
            'model__l2_regularization': [0.0, 0.1, 1.0],
        },
        cv=GroupKFold(n_splits=3),
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X_tr, y_tr, groups=grp_tr)

    return accuracy_score(y_te, grid.predict(X_te))

# COMPARISONS
results = {
    'Original':          evaluate(orig_feats),
    'Ratios+Interacts':  evaluate(eng_feats),
    'PCA-only':          evaluate(pca_feats),
    'Engineered+PCA':    evaluate(combo_feats),
    'Smart Recipe':      evaluate(smart_feats)
}
res_df = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy'])
print(res_df.sort_values('Accuracy', ascending=False))
