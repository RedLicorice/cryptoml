from statsmodels.tsa.seasonal import STL
import pandas as pd

def get_residual(s: pd.Series):
    res = STL(s).fit().resid
    return pd.Series(res, index=s.index)