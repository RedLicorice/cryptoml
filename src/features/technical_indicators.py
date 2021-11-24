import numpy as np
import pandas as pd
from pyti.exponential_moving_average import exponential_moving_average
from pyti.simple_moving_average import simple_moving_average
from pyti.moving_average_convergence_divergence import moving_average_convergence_divergence
from pyti.aroon import aroon_up, aroon_down, aroon_oscillator
from pyti.directional_indicators import (average_directional_index, positive_directional_index, negative_directional_index)
from pyti.price_oscillator import price_oscillator
from pyti.relative_strength_index import relative_strength_index
from pyti.money_flow_index import money_flow_index
from pyti.stochastic import percent_k
from pyti.chande_momentum_oscillator import chande_momentum_oscillator
from pyti.average_true_range_percent import average_true_range_percent
from pyti.volume_oscillator import volume_oscillator
from pyti.accumulation_distribution import accumulation_distribution
from pyti.on_balance_volume import on_balance_volume
from pyti.force_index import force_index
from pyti.true_strength_index import true_strength_index
from pyti.function_helper import fill_for_noncomputable_vals
from pyti.bollinger_bands import percent_b, upper_bollinger_band, middle_bollinger_band, lower_bollinger_band
from pyti.volatility import volatility
from pyti import catch_errors
import warnings
import logging


def relative_sma(data, short, long):
		sma_short = simple_moving_average(data, period=short)
		sma_long = simple_moving_average(data, period=long)
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", category=RuntimeWarning)
			smadiff = sma_short - sma_long
			rsma = np.divide(smadiff, sma_long)
		return fill_for_noncomputable_vals(data, rsma)

def relative_ema(data, short, long):
	ema_short = exponential_moving_average(data, period=short)
	ema_long = exponential_moving_average(data, period=long)
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", category=RuntimeWarning)
		emadiff = ema_short - ema_long
		rema = np.divide(emadiff, ema_long)
	return fill_for_noncomputable_vals(data, rema)

def percent_k_pr(high_data, low_data, close_data, period):
	"""
	%K.
	Formula:
	%k = data(t) - low(n) / (high(n) - low(n))
	"""
	# print (len(high_data))
	# print (period)
	catch_errors.check_for_period_error(high_data, period)
	catch_errors.check_for_period_error(low_data, period)
	catch_errors.check_for_period_error(close_data, period)
	percent_k = [((close_data[idx] - np.min(low_data[idx+1-period:idx+1])) /
		 (np.max(high_data[idx+1-period:idx+1]) -
		  np.min(low_data[idx+1-period:idx+1]))) for idx in range(period-1, len(close_data))]
	percent_k = fill_for_noncomputable_vals(close_data, percent_k)

	return percent_k

def _get_ta_features(high, low, close, volume, desc):
	"""
	Returns a dict containing the technical analysis indicators calculated on the given
	high, low, close and volumes.
	"""
	ta = {}

	# Set numpy to ignore division error and invalid values (since not all features are complete)
	old_settings = np.seterr(divide='ignore', invalid='ignore')
	record_count = len(close)

	# Determine relative moving averages
	for _short, _long in desc['rsma']:
		if record_count < _short or record_count < _long:
			logging.error("get_ta_features: not enough records for rsma (short={}, long={}, records={})"
						 .format(_short, _long, record_count))
			continue
		ta['rsma_{}_{}'.format(_short, _long)] = relative_sma(close, _short, _long)
	for _short, _long in desc['rema']:
		if record_count < _short or record_count < _long:
			logging.error("get_ta_features: not enough records for rema (short={}, long={}, records={})"
						 .format(_short, _long, record_count))
			continue
		ta['rema_{}_{}'.format(_short, _long)] = relative_ema(close, _short, _long)

	# MACD Indicator
	if 'macd' in desc:
		for _short, _long in desc['macd']:
			if record_count < _short or record_count < _long:
				logging.error("get_ta_features: not enough records for rema (short={}, long={}, records={})"
							 .format(_short, _long, record_count))
				continue
			ta['macd_{}_{}'.format(_short, _long)] = moving_average_convergence_divergence(close, _short, _long)

	# Aroon Indicator
	if 'ao' in desc:
		for _period in desc['ao']:
			if record_count < _period:
				logging.error("get_ta_features: not enough records for ao (period={}, records={})"
							 .format(_period, record_count))
				continue
			ta['ao_{}'.format(_period)] = aroon_oscillator(close, _period)

	# Average Directional Movement Index (ADX)
	if 'adx' in desc:
		for _period in desc['adx']:
			if record_count < _period:
				logging.error("get_ta_features: not enough records for adx (period={}, records={})"
							 .format(_period, record_count))
				continue
			ta['adx_{}'.format(_period)] = average_directional_index(close, high, low, _period)

	# Difference between Positive Directional Index(DI+) and Negative Directional Index(DI-)
	if 'wd' in desc:
		for _period in desc['wd']:
			if record_count < _period:
				logging.error("get_ta_features: not enough records for wd (period={}, records={})"
							 .format(_period, record_count))
				continue
			ta['wd_{}'.format(_period)] = \
				positive_directional_index(close, high, low, _period) \
				- negative_directional_index(close, high, low, _period)

	# Percentage Price Oscillator
	if 'ppo' in desc:
		for _short, _long in desc['ppo']:
			if record_count < _short or record_count < _long:
				logging.error("get_ta_features: not enough records for ppo (short={}, long={}, records={})"
							 .format(_short, _long, record_count))
				continue
			ta['ppo_{}_{}'.format(_short, _long)] = price_oscillator(close, _short, _long)

	# Relative Strength Index
	if 'rsi' in desc:
		for _period in desc['rsi']:
			if record_count < _period:
				logging.error("get_ta_features: not enough records for rsi (period={}, records={})"
							 .format(_period, record_count))
				continue
			ta['rsi_{}'.format(_period)] = relative_strength_index(close, _period)

	# Money Flow Index
	if 'mfi' in desc:
		for _period in desc['mfi']:
			if record_count < _period:
				logging.error("get_ta_features: not enough records for mfi (period={}, records={})"
							 .format(_period, record_count))
				continue
			ta['mfi_{}'.format(_period)] = money_flow_index(close, high, low, volume, _period)

	# True Strength Index
	if 'tsi' in desc and len(close) >= 40:
		if record_count < 40:
			logging.error("get_ta_features: not enough records for tsi (period={}, records={})"
						 .format(40, record_count))
		else:
			ta['tsi'] = true_strength_index(close)

	if 'boll' in desc:
		for _period in desc['stoch']:
			if record_count < _period:
				logging.error("get_ta_features: not enough records for boll (period={}, records={})"
							 .format(_period, record_count))
				continue
			ta['boll_{}'.format(_period)] = percent_b(close, _period)

	# Stochastic Oscillator
	if 'stoch' in desc:
		for _period in desc['stoch']:
			if record_count < _period:
				logging.error("get_ta_features: not enough records for stoch (period={}, records={})"
							 .format(_period, record_count))
				continue
			ta['stoch_{}'.format(_period)] = percent_k(close, _period)
	# ta.py['stoch'] = percent_k(high, low, close, 14)

	# Chande Momentum Oscillator
	## Not available in ta.py
	if 'cmo' in desc:
		for _period in desc['cmo']:
			if record_count < _period:
				logging.error("get_ta_features: not enough records for cmo (period={}, records={})"
							 .format(_period, record_count))
				continue
			ta['cmo_{}'.format(_period)] = chande_momentum_oscillator(close, _period)

	# Average True Range Percentage
	if 'atrp' in desc:
		for _period in desc['atrp']:
			if record_count < _period:
				logging.error("get_ta_features: not enough records for atrp (period={}, records={})"
							 .format(_period, record_count))
				continue
			ta['atrp_{}'.format(_period)] = average_true_range_percent(close, _period)

	# Percentage Volume Oscillator
	if 'pvo' in desc:
		for _short, _long in desc['pvo']:
			if record_count < _short or record_count < _long:
				logging.error("get_ta_features: not enough records for pvo (short={}, long={}, records={})"
							 .format(_short, _long, record_count))
				continue
			ta['pvo_{}_{}'.format(_short, _long)] = volume_oscillator(volume, _short, _long)

	# Force Index
	if 'fi' in desc:
		fi = force_index(close, volume)
		for _period in desc['fi']:
			if record_count < _period:
				logging.error("get_ta_features: not enough records for fi (period={}, records={})"
							 .format(_period, record_count))
				continue
			ta['fi_{}'.format(_period)] = exponential_moving_average(fi, _period)

	# Accumulation Distribution Line
	if 'adi' in desc:
		ta['adi'] = accumulation_distribution(close, high, low, volume)

	# On Balance Volume
	if 'obv' in desc:
		ta['obv'] = on_balance_volume(close, volume)

	# Restore numpy error settings
	np.seterr(**old_settings)

	return ta

def get_ta_features(ohlcv: pd.DataFrame, indicators: dict):
	ta = _get_ta_features(
		ohlcv['high'].values,
		ohlcv['low'].values,
		ohlcv['close'].values,
		ohlcv['volume'].values,
		indicators
	)

	result = pd.DataFrame(index=ohlcv.index)
	for k in ta.keys():  # Keys are the same both for 'ta' and 'dta'
		result[k] = ta[k]
	return result