import typer
import xgboost
app = typer.Typer()


@app.command(name='bootstrap', help='Bootstrap dataset with data from zip files in data/bootstrap')
def build_dataset(symbol: str, currency: str):
    target_name = '../data/dataset-{symbol}{currency}'.format(symbol=symbol, currency=currency)
    from crawlers import kraken, coinmetrics
    _kraken = kraken.get_bootstrap_data(symbol, currency).fillna(method='ffill')
    _coinmetrics = coinmetrics.get_bootstrap_data(symbol)

    ohlcv = kraken.ticks_to_ohlcv(_kraken, '1D').fillna(method='ffill')

    from dataset import build, get_feature_metadata, make_target
    import pandas as pd
    result = build(ohlcv=ohlcv, coinmetrics=_coinmetrics, W=10)
    result.to_csv(target_name + '.csv', index_label='timestamp')

    _begin, _end, _features = get_feature_metadata(result)
    meta = pd.DataFrame.from_records(_features)
    meta.index = meta['name']
    meta.drop(labels='name', axis='columns', inplace=True)
    meta.to_csv(target_name + '.meta.csv', index_label='feature')

    target = make_target(ohlcv)
    target.to_csv(target_name + '.target.csv', index_label='timestamp')

    info = {
        'symbol': symbol,
        'currency': currency,
        'interval': '1D',
        'records': result.shape[0],
        'features': result.shape[1],
        'index_min': result.index.min().to_pydatetime().isoformat(),
        'index_max': result.index.max().to_pydatetime().isoformat(),
        'valid_index_min': _begin,
        'valid_index_max': _end,
        'targets': {str(k): False if k != 'class' else True for k in target.columns},
        'features': {str(k): True for k in meta.index}
    }
    with open(target_name + '.info.yaml', 'w') as f:
        import yaml
        yaml.dump(info, f, sort_keys=False)
    print('done')

@app.command(name='selection', help='Perform feature selection and update <dataset>.info.yaml with selected features')
def selection(symbol: str, currency: str, percent: float):
    target_name = '../data/dataset-{symbol}{currency}'.format(symbol=symbol, currency=currency)
    import pandas as pd
    import math
    from crawlers import load_yaml, save_yaml

    info = load_yaml(target_name + '.info.yaml')
    dataset = pd.read_csv(target_name + '.csv', parse_dates=True, index_col='timestamp')
    target = pd.read_csv(target_name + '.target.csv', parse_dates=True, index_col='timestamp')
    first_valid_i = dataset.index.get_loc(info.valid_index_min)
    last_valid_i = dataset.index.get_loc(info.valid_index_max)

    training_records = math.floor((last_valid_i - first_valid_i) * percent)
    dataset['label'] = target['class']
    training_dataset = dataset.iloc[first_valid_i:first_valid_i+training_records]
    # testing_dataset = dataset.iloc[first_valid_i+training_records: last_valid_i+1]

    from xgboost import XGBClassifier
    from util.selection_pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    pipeline = Pipeline(steps=[
        ('s', StandardScaler()),
        ('i', SimpleImputer()),
        ('c', XGBClassifier(use_label_encoder=False))
    ])

    X_train = training_dataset.drop(labels=['label'], axis='columns')
    with pd.option_context('mode.use_inf_as_na', True):  # Set option temporarily
        X_train.fillna(axis='columns', method='ffill', inplace=True)
    y_train = training_dataset['label']

    from sklearn.feature_selection import SelectFromModel
    sel = SelectFromModel(pipeline)
    sel.fit(X_train, y_train)
    support = sel.get_support()

    dinfo = info.to_dict()
    for c, mask in zip(X_train.columns, support):
        dinfo['features'][c] = True if mask else False

    import yaml
    with open(target_name + '.info.yaml', 'w') as f:
        yaml.dump(dinfo, f, sort_keys=False)
    with open(target_name + '.info.yaml.bak', 'w') as f:
        yaml.dump(info, f, sort_keys=False)
    print('done')



@app.command()
def test(symbol: str, currency: str):
    from dataset import make_ohlcv_ta
    from crawlers import kraken
    _kraken = kraken.get_bootstrap_data(symbol, currency).fillna(method='ffill')
    ohlcv = kraken.ticks_to_ohlcv(_kraken, '1D').fillna(method='ffill')
    ohlcv_ta = make_ohlcv_ta(ohlcv)
    print('It works')
    print(ohlcv_ta.head())

if __name__ == '__main__':
    app()