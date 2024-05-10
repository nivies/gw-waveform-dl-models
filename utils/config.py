import json
from dotmap import DotMap
import os
import time


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = DotMap(config_dict)

    return config, config_dict


def process_config(json_file):
    config, config_dict = get_config_from_json(json_file)
    config.callbacks.tensorboard_log_dir = os.path.join("experiments", time.strftime("%Y-%m-%d_%H-%M-%S/",time.localtime()), config.exp.name, "logs/")
    config.callbacks.checkpoint_dir = os.path.join("experiments", time.strftime("%Y-%m-%d_%H-%M-%S/",time.localtime()), config.exp.name, "checkpoints/")
    return config, config_dict

def init_obj(config, part, module, *args, **kwargs):
    """
    Finds the function handle for the code part (trainer/data_loader/model) in config, and returns the
    instance initialized with corresponding arguments given.

    `object = init_obj(config, 'name', module, a, b=1)`
    is equivalent to
    `object = module.name(a, b=1)`
    """
    module_name = config[part]['name']
    module_args = dict(config[part]['args'])
    assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    return getattr(module, module_name)(config, *args, **module_args)
