import os


# Default data directory (/path/pycls/pycls/datasets/data)
_DEF_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Data paths
_paths = {
    'cifar10': _DEF_DATA_DIR + '/cifar10',
    'imagenet': _DEF_DATA_DIR + '/imagenet',
}


def has_data_path(dataset_name):
    '''Determines if the dataset has a data path.'''
    return dataset_name in _paths.keys()


def get_data_path(dataset_name):
    '''Retrieves data path for the dataset.'''
    return _paths[dataset_name]


def register_path(name, path):
    '''Registers a dataset path dynamically.'''
    _paths[name] = path
