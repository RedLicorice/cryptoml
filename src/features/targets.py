import pandas as pd
import numpy as np
from util.discretization import to_discrete_double, to_discrete_single
from sklearn.preprocessing import KBinsDiscretizer

def target_price(close : pd.Series, **kwargs):
    return close.shift(-kwargs.get('periods', 1)).fillna(method='ffill')

def target_pct(close : pd.Series, **kwargs):
    pct_var = pd.Series(np.roll(close.pct_change(periods=kwargs.get('periods', 1)), -kwargs.get('periods', 1)), index=close.index).fillna(method='ffill')
    return pct_var

def target_class(close : pd.Series, **kwargs):
    pct_var = target_pct(close, **kwargs)
    classes = to_discrete_double(pct_var.fillna(method='ffill'), -0.01, 0.01)
    return pd.Series(classes, index=pct_var.index)

def target_binary(close : pd.Series, **kwargs):
    pct_var = target_pct(close, **kwargs)
    classes = to_discrete_single(pct_var.fillna(method='ffill'), 0.00)
    return pd.Series(classes, index=pct_var.index)

def target_label(classes, **kwargs):
    _labels = ['SELL', 'HOLD', 'BUY']
    if 'labels' in kwargs:
        _labels = kwargs.get('labels')
    return pd.Series([_labels[int(c)] for c in classes], index=classes.index)

def target_binned_class(close : pd.Series, **kwargs):
    pct_var = target_pct(close, **kwargs)
    values = pct_var.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').values
    values = np.reshape(values, (-1, 1))
    discretizer = KBinsDiscretizer(n_bins=kwargs.get('n_bins',3), strategy='quantile', encode='ordinal')
    discrete = discretizer.fit_transform(values)
    return pd.Series(np.reshape(discrete, (-1,)), index=pct_var.index)

def target_binned_class_kmeans(close : pd.Series, **kwargs):
    pct_var = target_pct(close, **kwargs)
    values = pct_var.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').values
    values = np.reshape(values, (-1, 1))
    discretizer = KBinsDiscretizer(n_bins=kwargs.get('n_bins',3), strategy='kmeans', encode='ordinal')
    discrete = discretizer.fit_transform(values)
    return pd.Series(np.reshape(discrete, (-1,)), index=pct_var.index)