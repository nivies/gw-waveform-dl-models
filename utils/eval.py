import os
from utils.config import process_config, init_obj
import data_loader.gw_dataloader as data_loader_module
import models.gw_models as models_module

def load_model_and_data_loader(model_dir):

    config, _ = process_config(os.path.join(model_dir, "config.json"))

    print("Loading data...", end="\r")
    data_loader = init_obj(config, "data_loader", data_loader_module)

    if config.model.name == "RegularizedAutoEncoderGenerator":
        model = init_obj(config, "model", models_module, data_loader = data_loader, inference = True)
    else:
        model = init_obj(config, "model", models_module, data_loader = data_loader)
    print("Loading model...     ", end="\r")
    model.load(os.path.join(model_dir, "best_model.hdf5"))

    return model.model, data_loader