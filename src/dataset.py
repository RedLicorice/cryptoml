import pandas as pd

TA_CONFIG = {
    'rsma': [(5, 20), (8, 15), (20, 50)],
    'rema': [(5, 20), (8, 15), (20, 50)],
    'macd': [(12, 26)],
    'ao': [14],
    'adx': [14],
    'wd': [14],
    'ppo': [(12, 26)],
    'rsi': [14],
    'mfi': [14],
    'tsi': None,
    'stoch': [14],
    'cmo': [14],
    'atrp': [14],
    'pvo': [(12, 26)],
    'fi': [13, 50],
    'adi': None,
    'obv': None
}


def get_feature_metadata(df):
    feature_indices = []
    global_first = None
    global_last = None
    for c in df.columns:
        # If all elements are non-NA/null, first_valid_index and last_valid_index return None.
        # They also return None for empty Series/DataFrame.
        if df[c].empty:
            print("Feature {} is empty!".format(c))
            continue
        fvi = df[c].first_valid_index() or df[c].index.min()
        lvi = df[c].last_valid_index() or df[c].index.max()
        _first = fvi.to_pydatetime()
        _last = lvi.to_pydatetime()
        feature_indices.append({
            'name': str(c),
            'first': _first.isoformat(),
            'last': _last.isoformat(),
            'count': df[c].shape[0],
            'null': df[c].isna().sum(),
            'distinct': df[c].value_counts().sum()
        })
        if not global_first or _first > global_first:
            global_first = _first
        if not global_last or _last < global_last:
            global_last = _last
    return global_first.isoformat(), global_last.isoformat(), feature_indices


def make_ohlcv_lags(ohlcv, W):
    from features.lagging import make_lagged
    lagged_ohlcv = pd.concat(
        [make_lagged(ohlcv, i) for i in range(1, W + 1)],
        axis='columns',
        verify_integrity=True,
        sort=True,
        join='inner'
    )
    return lagged_ohlcv


def make_ohlcv_pct(ohlcv):
    ohlcv_pct = ohlcv[['open', 'high', 'low', 'close', 'volume']].pct_change()
    ohlcv_pct.columns = ['{}_pct'.format(c) for c in ohlcv_pct.columns]
    return ohlcv_pct


def make_ohlc_patterns(ohlcv):
    from features.talib import get_talib_patterns
    _patterns = get_talib_patterns(ohlcv)
    ohlc_patterns = pd.DataFrame(index=ohlcv.index)
    ohlc_patterns['talib_patterns_mean'] = _patterns.mean(axis=1)
    ohlc_patterns['talib_patterns_sum'] = _patterns.sum(axis=1)
    return ohlc_patterns


def make_ohlc_residual(ohlcv):
    from features.decompose import get_residual
    # Residual from STL Decomposition of OHLC data
    ohlc_residuals = pd.DataFrame()
    ohlc_residuals['open_resid'] = get_residual(ohlcv.open)
    ohlc_residuals['high_resid'] = get_residual(ohlcv.high)
    ohlc_residuals['low_resid'] = get_residual(ohlcv.low)
    ohlc_residuals['close_resid'] = get_residual(ohlcv.close)
    return ohlc_residuals


def make_ohlc_splines(ohlcv):
    from features.spline import get_spline
    # SPLINES
    # Use SPLINES to extract price information
    ohlc_splines = pd.DataFrame(index=ohlcv.index)
    # First derivative indicates slope
    ohlc_splines['open_spl_d1'] = get_spline(ohlcv.open, 1)
    ohlc_splines['high_spl_d1'] = get_spline(ohlcv.high, 1)
    ohlc_splines['low_spl_d1'] = get_spline(ohlcv.low, 1)
    ohlc_splines['close_spl_d1'] = get_spline(ohlcv.close, 1)
    # Second derivative indicates convexity
    ohlc_splines['open_spl_d2'] = get_spline(ohlcv.open, 2)
    ohlc_splines['high_spl_d2'] = get_spline(ohlcv.high, 2)
    ohlc_splines['low_spl_d2'] = get_spline(ohlcv.low, 2)
    ohlc_splines['close_spl_d2'] = get_spline(ohlcv.close, 2)
    return ohlc_splines


def make_ohlcv_stats(ohlcv):
    from features.ohlcv import ohlcv_resample

    # Relevant stats from OHLC data interpretation
    ohlcv_stats = pd.DataFrame(index=ohlcv.index)
    ohlcv_stats['close_open_pct'] = (ohlcv.close - ohlcv.open).pct_change()  # Change in body of the candle (> 0 if candle is green)
    ohlcv_stats['high_close_dist_pct'] = (ohlcv.high - ohlcv.close).pct_change()  # Change in wick size of the candle, shorter wick should be bullish
    ohlcv_stats['low_close_dist_pct'] = (ohlcv.close - ohlcv.low).pct_change()  # Change in shadow size of the candle, this increasing would indicate support (maybe a bounce)
    ohlcv_stats['high_low_dist_pct'] = (ohlcv.high - ohlcv.low).pct_change()  # Change in total candle size, smaller candles stands for low volatility
    ohlcv_stats['close_volatility_3d'] = ohlcv.close.pct_change().rolling(3).std(ddof=0)
    ohlcv_stats['close_volatility_7d'] = ohlcv.close.pct_change().rolling(7).std(ddof=0)
    ohlcv_stats['close_volatility_30d'] = ohlcv.close.pct_change().rolling(30).std(ddof=0)


    # Stats from resampled OHLC data
    for d in [3, 7, 30]:
        ohlcv_d = ohlcv_resample(ohlcv=ohlcv, period=d, interval='D')
        ohlcv_stats['close_open_pct_d{}'.format(d)] = (ohlcv_d.close - ohlcv_d.open).pct_change()
        ohlcv_stats['high_close_dist_pct_d{}'.format(d)] = (ohlcv_d.high - ohlcv_d.close).pct_change()
        ohlcv_stats['low_close_dist_pct_d{}'.format(d)] = (ohlcv_d.close - ohlcv_d.low).pct_change()
        ohlcv_stats['high_low_dist_pct_d{}'.format(d)] = (ohlcv_d.high - ohlcv_d.low).pct_change()

    return ohlcv_stats


def make_ohlcv_ta(ohlcv):
    import talib as ta
    result = pd.DataFrame(index=ohlcv.index)
    result['adx'] = ta.ADX(ohlcv.high, ohlcv.low, ohlcv.close, timeperiod=14)
    result['adxr'] = ta.ADXR(ohlcv.high, ohlcv.low, ohlcv.close, timeperiod=14)
    result['apo'] = ta.APO(ohlcv.close, fastperiod=12, slowperiod=26)
    aroon_down, aroon_up = ta.AROON(ohlcv.high, ohlcv.low, timeperiod=14)
    result['aroon_down'] = aroon_down
    result['aroon_up'] = aroon_up
    result['aroonosc'] = ta.AROONOSC(ohlcv.high, ohlcv.low, timeperiod=14)
    result['bop'] = ta.BOP(ohlcv.open, ohlcv.high, ohlcv.low, ohlcv.close)
    result['cmo'] = ta.CMO(ohlcv.close, timeperiod=14)
    result['cci'] = ta.CCI(ohlcv.high, ohlcv.low, ohlcv.close, timeperiod=14)
    result['dx'] = ta.DX(ohlcv.high, ohlcv.low, ohlcv.close, timeperiod=14)
    macd, macdsignal, macdhist = ta.MACD(ohlcv.close, fastperiod=12, slowperiod=26, signalperiod=9)
    result['macd'] = macd
    result['macdsignal'] = macdsignal
    result['macdhist'] = macdhist
    result['mfi'] = ta.MFI(ohlcv.high, ohlcv.low, ohlcv.close, ohlcv.volume, timeperiod=14)
    result['minus_di'] = ta.MINUS_DI(ohlcv.high, ohlcv.low, ohlcv.close, timeperiod=14)
    result['minus_dm'] = ta.MINUS_DM(ohlcv.high, ohlcv.low, timeperiod=14)
    result['mom'] = ta.MOM(ohlcv.close, timeperiod=10)
    result['plus_di'] = ta.MINUS_DI(ohlcv.high, ohlcv.low, ohlcv.close, timeperiod=14)
    result['plus_dm'] = ta.MINUS_DM(ohlcv.high, ohlcv.low, timeperiod=14)
    result['ppo'] = ta.PPO(ohlcv.close, fastperiod=12, slowperiod=26, matype=0)
    result['roc'] = ta.ROC(ohlcv.close, timeperiod=10)
    result['rocp'] = ta.ROCP(ohlcv.close, timeperiod=10)
    result['rocr'] = ta.ROCR(ohlcv.close, timeperiod=10)
    result['rocr100'] = ta.ROCR100(ohlcv.close, timeperiod=10)
    result['rsi'] = ta.RSI(ohlcv.close, timeperiod=14)
    slowk, slowd = ta.STOCH(ohlcv.high, ohlcv.low, ohlcv.close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    result['stoch_slowk'] = slowk
    result['stoch_slowd'] = slowd
    fastk, fastd = ta.STOCHF(ohlcv.high, ohlcv.low, ohlcv.close, fastk_period=5, fastd_period=3, fastd_matype=0)
    result['stochf_fastk'] = fastk
    result['stochf_fastd'] = fastd
    rfastk, rfastd = ta.STOCHRSI(ohlcv.close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    result['stochrsi_fastk'] = rfastk
    result['stochrsi_fastd'] = rfastd
    result['trix'] = ta.TRIX(ohlcv.close, timeperiod=30)
    result['ultosc'] = ta.ULTOSC(ohlcv.high, ohlcv.low, ohlcv.close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    result['willr'] = ta.WILLR(ohlcv.high, ohlcv.low, ohlcv.close, timeperiod=14)
    result['atr'] = ta.ATR(ohlcv.high, ohlcv.low, ohlcv.close, timeperiod=14)
    result['natr'] = ta.NATR(ohlcv.high, ohlcv.low, ohlcv.close, timeperiod=14)
    result['trange'] = ta.TRANGE(ohlcv.high, ohlcv.low, ohlcv.close)
    result['ad'] = ta.AD(ohlcv.high, ohlcv.low, ohlcv.close, ohlcv.volume)
    result['adosc'] = ta.ADOSC(ohlcv.high, ohlcv.low, ohlcv.close, ohlcv.volume, fastperiod=3, slowperiod=10)
    result['obv'] = ta.OBV(ohlcv.close, ohlcv.volume)
    # Missing functions from PyTi
    # Missing only RSMA, REMA, TSI, PVO, FI, ADI
    from pyti.force_index import force_index
    from pyti.true_strength_index import true_strength_index
    from pyti.volume_oscillator import volume_oscillator
    from pyti.accumulation_distribution import accumulation_distribution
    from features.technical_indicators import relative_sma, relative_ema
    result['fi'] = force_index(ohlcv.close, ohlcv.volume)
    result['tsi'] = true_strength_index(ohlcv.close)
    result['pvo_12_26'] = volume_oscillator(ohlcv.volume, short_period=12, long_period=26)
    result['adi'] = accumulation_distribution(ohlcv.close, ohlcv.high, ohlcv.low, ohlcv.volume)
    result['rsma_3_7'] = relative_sma(ohlcv.close, short=3, long=7)
    result['rema_3_7'] = relative_ema(ohlcv.close, short=3, long=7)
    result['rsma_12_26'] = relative_sma(ohlcv.close, short=12, long=26)
    result['rema_12_26'] = relative_ema(ohlcv.close, short=12, long=26)
    result['rsma_24_50'] = relative_sma(ohlcv.close, short=24, long=50)
    result['rema_24_50'] = relative_ema(ohlcv.close, short=24, long=50)
    return result


def make_target(ohlcv):
    from features.targets import target_price, target_pct, target_class, target_binary, target_binned_class
    result = pd.DataFrame(index=ohlcv.index)
    result['price'] = target_price(ohlcv.close)
    result['pct'] = target_pct(ohlcv.close)
    result['class'] = target_class(ohlcv.close)
    result['binary'] = target_binary(ohlcv.close)
    result['bin_class'] = target_binned_class(ohlcv.close, n_bins=3)
    result['bin_binary'] = target_binned_class(ohlcv.close, n_bins=2)
    return result


def build(ohlcv: pd.DataFrame, coinmetrics: pd.DataFrame, **kwargs):
    W = kwargs.get('W', 10)
    # ATSA - OHLC with 10-lag + TA
    lagged_ohlcv = make_ohlcv_lags(ohlcv, W)

    # Lagged percent variation of OHLCV
    ohlcv_pct = make_ohlcv_pct(ohlcv)
    lagged_ohlcv_pct = make_ohlcv_lags(ohlcv_pct, W)
    ohlc_patterns = make_ohlc_patterns(ohlcv)
    # ohlc_splines = make_ohlc_splines(ohlcv)
    ohlc_residuals = make_ohlc_residual(ohlcv)
    lagged_ohlc_residuals = make_ohlcv_lags(ohlc_residuals, W=10)
    ohlcv_stats = make_ohlcv_stats(ohlcv)
    # from features.technical_indicators import get_ta_features
    # ta = get_ta_features(ohlcv, TA_CONFIG)
    ta = make_ohlcv_ta(ohlcv)

    cm_percent = coinmetrics.pct_change(periods=1, fill_method='ffill')
    cm_percent.columns = [c+'_pct' for c in cm_percent.columns]

    merge_dataframes = [
        ohlcv, lagged_ohlcv,
        ohlcv_pct, lagged_ohlcv_pct,
        ohlc_residuals, lagged_ohlc_residuals,
        ohlcv_stats,
        ohlc_patterns,
        ta,
        coinmetrics, cm_percent
    ]

    # Drop columns whose values are all nan or inf from each facet
    with pd.option_context('mode.use_inf_as_na', True):  # Set option temporarily
        for _df in merge_dataframes:
            _df.dropna(axis='columns', how='all', inplace=True)
    return pd.concat(
        merge_dataframes,
        axis='columns',
        verify_integrity=True,
        sort=True,
        join='outer'
    )
