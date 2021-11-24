import pandas as pd
from typing import Union

def make_lagged(df: Union[pd.Series, pd.DataFrame], periods=1):
	shift = df.shift(periods=periods)
	if hasattr(shift, 'columns'):
		shift.columns = ['{}_lag{}'.format(c, periods) for c in shift.columns]
	elif hasattr(shift, 'name'):
		shift.rename('{}_lag{}'.format(shift.name, periods))
	return shift