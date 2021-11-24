def api():
    from . import load_api_spec
    return load_api_spec('crawlers/kraken.yaml')

def get_pair_ohlc(pair, metrics, frequency, begin, end):
    global kraken
    if type(metrics) is list:
        metrics = ','.join(metrics)
    import requests
    resp = requests.get(api().ohlc_data(assets=pair, metrics=metrics, frequency=frequency, since=0))
    rj = resp.json()
    result = rj['data']
    import time
    if 'next_page_url' in rj:
        while True:
            time.sleep(0.5)
            resp = requests.get(rj['next_page_url'])
            rj = resp.json()
            result += rj['data']
            if not 'next_page_url' in rj:
                break
    return result


def get_bootstrap_data(symbol, currency):
    from . import bootstrap_index, load_transformer
    _convert_map = {
        'btc':'xbt',
        'doge':'xdg'
    }
    if symbol in _convert_map:
        symbol = _convert_map[symbol]
    if currency in _convert_map:
        currency = _convert_map[currency]
    try:
        index = bootstrap_index('../data/bootstrap/index.yaml')
        transformer = load_transformer('../data/bootstrap/' + index.kraken.transformer)
        if index.kraken.groups:
            raise ValueError('Groups are not supported for kraken Loader')
        filename = index.kraken.name_format.format(symbol=symbol.upper(), currency=currency.upper()) + '.csv'
        return transformer.get_df('../data/bootstrap/' + index.kraken.zipfile, filename)
    except Exception as e:
        print('Exception occurred!    ' + str(e))
        raise


def ticks_to_ohlcv(ticks, interval):
    resample = ticks.resample(interval)
    ohlc = resample['price'].ohlc()
    ohlc['volume'] = resample['amount'].sum()
    return ohlc