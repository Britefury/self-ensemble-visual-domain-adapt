import numpy as np
from sklearn.decomposition import PCA


def standardise_samples(X):
    X = X - X.mean(axis=(1,2,3), keepdims=True)
    X = X / X.std(axis=(1,2,3), keepdims=True)
    return X

def standardise_dataset(ds):
    ds.train_X_orig = ds.train_X[...].copy()
    ds.val_X_orig = ds.val_X[...].copy()
    ds.test_X_orig = ds.test_X[...].copy()
    ds.train_X = standardise_samples(ds.train_X[...])
    ds.val_X = standardise_samples(ds.val_X[...])
    ds.test_X = standardise_samples(ds.test_X[...])
