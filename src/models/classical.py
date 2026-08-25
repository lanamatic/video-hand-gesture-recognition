import os
import time
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from src.evaluation.metrics import measure_inference_time

# Measure training time (single fresh fit of a cloned estimator) and
# per-sample inference time (100 runs on one test sample).
def measure_train_and_inference_time(estimator, X_train, y_train, X_test):

    model = clone(estimator)

    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time_s = time.perf_counter() - start

    inf = measure_inference_time(model.predict, X_test[:1], n_runs=100)

    return {
        'train_time_s': train_time_s,
        'inference_mean_ms': inf['mean_ms'],
        'inference_std_ms': inf['std_ms'],
        'inference_fps': inf['fps'],
    }

# Load all .npy landmark sequences listed in metadata, and aggregate each sequence (T, 63)
# into a fixed-length feature vector; return features and labels
def load_features_and_labels(metadata_path):

    meta = pd.read_csv(metadata_path)
    meta_dir = os.path.dirname(os.path.abspath(metadata_path))

    features = []
    labels = []

    for _, row in meta.iterrows():
        npy_path = row['npy_path']
        if not os.path.isabs(npy_path) and not os.path.exists(npy_path):
            # npy_path in metadata was saved relative to the project root
            # resolve relative to the metadata file's own directory instead
            npy_path = os.path.join(meta_dir, os.path.basename(npy_path))

        seq = np.load(npy_path)  # shape (T, 63)

        feat = np.concatenate([
            seq.mean(axis=0),
            seq.std(axis=0),
            seq.min(axis=0),
            seq.max(axis=0),
            np.diff(seq, axis=0).mean(axis=0),
            np.diff(seq, axis=0).std(axis=0)
        ])

        features.append(feat)
        labels.append(row['label'])

    X = np.array(features)
    y = np.array(labels)

    return X, y

# Grid search over k and distance metric for KNN
def knn(X_train, y_train):

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier()),
    ])

    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9, 11, 15, 21],
        'knn__metric': ['euclidean', 'manhattan', 'minkowski'],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = GridSearchCV(pipe, param_grid, scoring='f1_weighted', cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)

    print(f"Best params: {search.best_params_}")
    print(f"Best CV F1 (weighted): {search.best_score_:.4f}")

    return search

# Grid search over C for linear SVM
def svm_linear(X_train, y_train):

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='linear', probability=True)),
    ])

    param_grid = {
        'svm__C': [0.01, 0.1, 1, 10, 100],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = GridSearchCV(pipe, param_grid, scoring='f1_weighted', cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)

    print(f"Best params: {search.best_params_}")
    print(f"Best CV F1 (weighted): {search.best_score_:.4f}")

    return search

# Grid search over C and gamma for RBF-kernel SVM
def svm_rbf(X_train, y_train):

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True)),
    ])

    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': [0.001, 0.01, 0.1, 1, 'scale'],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = GridSearchCV(pipe, param_grid, scoring='f1_weighted', cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)

    print(f"Best params: {search.best_params_}")
    print(f"Best CV F1 (weighted): {search.best_score_:.4f}")

    return search

# Grid search over n_estimators, max_depth, min_samples_split for Random Forest
def random_forest(X_train, y_train):

    param_grid = {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, scoring='f1_weighted', cv=cv, n_jobs=-1
    )
    search.fit(X_train, y_train)

    print(f"Best params: {search.best_params_}")
    print(f"Best CV F1 (weighted): {search.best_score_:.4f}")

    return search

# Recursively convert numpy types/arrays to plain Python types for JSON
def _to_serializable(obj):

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj

# Save a fitted estimator (joblib) and its evaluation results + best hyperparameter config (JSON)
def save_model_and_results(model_name, estimator, results, best_params,
                            save_dir="../results/classical_models"):

    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, f"{model_name}.joblib")
    joblib.dump(estimator, model_path)

    results_with_config = dict(results)
    results_with_config['config'] = {
        'best_params': best_params,
        'cv_folds': 5,
        'scoring': 'f1_weighted',
    }

    results_path = os.path.join(save_dir, f"{model_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(_to_serializable(results_with_config), f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved results to {results_path}")