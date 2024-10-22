import tensorflow as tf
import argparse
from data_loader import gw_dataloader as data_loader_module
from models import gw_models as models_module
from utils.plot_utils import make_plots
from utils.config import process_config, init_obj
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def load_model_and_data_loader(model_dir):

    try:

        config, _ = process_config(os.path.join(model_dir, "config.json"))
    except:

        config, _ = process_config(os.path.join(os.path.dirname(model_dir), "config.json"))

    print("Loading data...", end="\r")
    data_loader = init_obj(config, "data_loader", data_loader_module)

    if config.model.name == "RegularizedAutoEncoderGenerator":
        model = init_obj(config, "model", models_module, data_loader = data_loader, inference = True)
    else:
        model = init_obj(config, "model", models_module, data_loader = data_loader)

    print("Loading model...     ", end="\r")
    model.autoencoder.autoencoder.load_weights(os.path.join(model_dir, "best_autoencoder.hdf5"))

    return model.autoencoder.autoencoder, data_loader, config

print("\n")

def main():

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-l', '--load_checkpoint', dest='load_dir', metavar='LC', default='None', help='Directory of checkpoint to load model')
    args = argparser.parse_args()
    model, data_loader, config = load_model_and_data_loader(args.load_dir) 

    data_loader.X_train = data_loader.y_train
    data_loader.X_test = data_loader.y_test

    opt_dir = os.path.join(os.path.dirname(args.load_dir), "ae_figures")

    print("Making plots")
    config.trainer.uninitialised = True # Bad hotfix
    make_plots(model = model, dir = opt_dir, data_loader = data_loader, config = config, metric = "overlap")


if __name__ == '__main__':
    main()
