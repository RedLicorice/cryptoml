import confuse
from dotenv import load_dotenv
import re, os, yaml

# Load Configuration
env_variable_pattern = re.compile('.*?\${(\w+)}.*?')
_configfile = None
load_dotenv()

def replace_env_variables(loader, node):
    """
    Extracts the environment variable from the node's value
    :param yaml.Loader loader: the yaml loader
    :param node: the current node in the yaml
    :return: the parsed string that contains the value of the environment
    variable
    """
    value = loader.construct_scalar(node)
    match = env_variable_pattern.findall(value)  # to find all env variables in line
    if match:
        full_value = value
        for g in match:
            full_value = full_value.replace(
                f'${{{g}}}', os.environ.get(g, g)
            )
        return full_value if not full_value.isnumeric() else int(full_value)
    return value

confuse.Loader.add_constructor('!ENV', replace_env_variables)
#config = confuse.Configuration('CryptoML-API', __name__)
config = confuse.LazyConfig('CryptoML', __name__)
config.set_file('../config.yaml')
