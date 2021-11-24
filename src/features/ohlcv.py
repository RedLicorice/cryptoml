import pandas as pd


def ohlcv_from_ticks(price: pd.Series, volume: pd.Series, interval, **kwargs):
    ohlc = price.resample(interval).ohlc()
    volume = volume.resample(interval).sum()
    volume.name = 'volume'
    ohlcv = pd.concat([ohlc, volume], axis='columns')
    return ohlcv


def ohlcv_resample(ohlcv: pd.DataFrame, **kwargs):
    period = int(kwargs.get('period', 7))
    interval = kwargs.get('interval', 'D')
    process_fun = kwargs.get('process_fun', lambda x: x)
    rename_fun = kwargs.get('rename_fun', None)
    result = []
    df = ohlcv.sort_index()
    for i in range(period):
        _df = df.iloc[i:]
        nth_day = _df.resample('{}{}'.format(period, interval),
                               closed='left',
                               label='right',
                               convention='end',
                               kind='timestamp'
                ).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).copy()

        result.append(process_fun(nth_day))
    _result = pd.concat(result, sort=True).sort_index()
    if rename_fun:
        _result.columns = rename_fun([c for c in _result.columns])
    if kwargs.get('trim', True):
        _result = _result.loc[ohlcv.first_valid_index():ohlcv.last_valid_index()]
    return _result