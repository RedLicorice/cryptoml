import yaml
from util.bunch import Bunch


class Spec(dict):
    def __init__(self, base_url, endpoints, **kwargs):
        if not type(endpoints) is dict:
            raise ValueError('Endpoints must be a dictionary!')
        self.base_url = base_url
        self.endpoints = endpoints
        super().__init__(kwargs)

    @staticmethod
    def from_dict(spec):
        if not spec:
            raise ValueError('Provided spec is invalid.')
        if not 'base_url' in spec:
            raise ValueError('Provided spec does not describe a base url.')
        res = Spec(base_url=spec['base_url'], endpoints=spec['endpoints'])
        return res

    def __dir__(self):
        return self.endpoints.keys()

    def __setstate__(self, state):
        pass

    def __setattr__(self, key, value):
        if key in ['endpoints', 'base_url']:
            self[key] = value
        else:
            self.endpoints[key] = value

    def __getattr__(self, key):
        if key in ['endpoints', 'base_url']:
            return self[key]
        try:
            query = self.endpoints[key]
            if '{' in query and '}' in query:
                return (self.base_url + query).format
            return self.base_url + query
        except KeyError:
            raise AttributeError('Method not described in api spec: ' + key)


def load_api_spec(filename):
    with open(filename, 'r') as f:
        try:
            spec = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print('error loading api spec' + exc)
    if spec:
        return Spec.from_dict(spec)

def load_yaml(filename):
    with open(filename, 'r') as f:
        try:
            spec = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print('error loading yaml' + exc)
    if spec:
        return Bunch(**spec)

def save_yaml(filename, data):
    with open(filename, 'w') as f:
        try:
            yaml.safe_dump(data=data, stream=f)
        except yaml.YAMLError as exc:
            print('error saving yaml' + exc.message or str(exc))

def load_bootstrap(zip_file, csv_file):
    import zipfile
    import pandas as pd
    with zipfile.ZipFile(zip_file) as z:
        with z.open(csv_file) as f:
            train = pd.read_csv(f, delimiter=",", parse_dates=True, index_col='date')
            return train

def bootstrap_index(filename):
    index = load_yaml(filename)
    return index.bootstrap

def load_transformer(filename):
    import importlib
    import ntpath
    
    spec = importlib.util.spec_from_file_location("bootstrap."+ntpath.basename(filename)[:-3], filename)
    transformer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transformer)
    return transformer