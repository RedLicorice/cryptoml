def api():
    from . import load_api_spec
    return load_api_spec('crawlers/coinmetrics-community.yaml')

def get_assets(assets):
    if type(assets) is list:
        assets = ','.join(assets)
    import requests
    resp = requests.get(api().asset_metadata(assets=assets))
    return resp.json()['data']

def get_asset_features(asset_name, frequency):
    assets = get_assets(asset_name)
    result = []
    id = 1
    for asset in assets:
        for metric in asset['metrics']:
            for f in metric['frequencies']:
                if frequency == f['frequency']:
                    result.append({'index': id, 'dataset': 'coinmetrics', 'asset': asset['asset'], 'name': metric['metric'], 'min': f['min_time'], 'max': f['max_time'], 'enabled': True})
                    id += 1
    return result

def get_asset_metrics(asset_name, metrics, frequency, begin, end):
    if type(metrics) is list:
        metrics = ','.join(metrics)
    import requests
    resp = requests.get(api().metrics_timeseries(assets=asset_name, metrics=metrics, frequency=frequency, begin=begin, end=end))
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


def get_bootstrap_data(symbol):
    symbol = symbol.lower()

    from . import bootstrap_index, load_transformer
    try:
        index = bootstrap_index('../data/bootstrap/index.yaml')
        transformer = load_transformer('../data/bootstrap/' + index.coinmetrics.transformer)
        if symbol not in index.coinmetrics.groups:
            filename = index.coinmetrics.name_format.format(symbol=symbol) + '.csv'
            return transformer.get_df('../data/bootstrap/' + index.coinmetrics.zipfile, filename)
        else:
            filenames = [index.coinmetrics.name_format.format(symbol=symbol) + '.csv']
            filenames += [ name + '.csv' for name in index.coinmetrics.groups[symbol]]
            dataframes = [transformer.get_df('../data/bootstrap/' + index.coinmetrics.zipfile, filename) for filename in filenames]

            import pandas as pd
            return pd.concat(dataframes)
    except Exception as e:
        print('Exception occurred!    ' + str(e))
        raise
