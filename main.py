import json
import os
from joblib import dump
import tensorflow as tf
import data_loader.gw_dataloader as data_loader_module
import models.gw_models as models_module
import trainers.gw_trainer as trainers_module
from utils.config import process_config, init_obj
from utils.dirs import create_dirs
from utils.utils import get_args
from utils.plot_utils import make_plots

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config, config_dict = process_config(args.config)
    except:
        print("Missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])
    with open(os.path.join(config.callbacks.checkpoint_dir, "config.json"), 'w') as fp:
        json.dump(config_dict, fp)

    print("\nLoading data...", end = "\r")
    data_loader = init_obj(config, "data_loader", data_loader_module)
    print("Data loaded!")
    print("Loading model...", end="\r")

    if config.trainer.name == "GWModelTrainer" and config.model.name == "RegularizedAutoEncoderGenerator":
        model = init_obj(config, "model", models_module, data_loader = data_loader, inference = True)
    else:
        model = init_obj(config, "model", models_module, data_loader = data_loader)
    
    print("Model loaded!")
    print("Initialising training...")
    trainer = init_obj(config, "trainer", trainers_module, model = model, data = data_loader)

    trainer.train()

    root_dir = os.path.dirname(config.callbacks.checkpoint_dir)
    fig_directory = os.path.join(os.path.dirname(root_dir), "figures")

    make_plots(model = model.model, dir = fig_directory, data_loader = data_loader, config = config, metric = 'overlap')


if __name__ == '__main__':
    main()
