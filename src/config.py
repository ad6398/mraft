import os, yaml


def get_config(file_path):
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config

def get_colpali_config():
    COLPALI_CONFIG_PATH = 'configs/colpali_sft.yaml'
    return get_config(COLPALI_CONFIG_PATH)





