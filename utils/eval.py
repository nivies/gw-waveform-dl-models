import os
from utils.config import process_config, init_obj
import data_loader.gw_dataloader as data_loader_module
import models.gw_models as models_module

def load_model_and_data_loader(model_dir):

    try:

        config, _ = process_config(os.path.join(model_dir, "config.json"))
    except:

        config, _ = process_config(os.path.join(os.path.dirname(model_dir), "config.json"))

    print("Loading data...", end="\r")
    data_loader = init_obj(config, "data_loader", data_loader_module)

    if config.model.name == "RegularizedAutoEncoderGenerator":
        model = init_obj(config, "model", models_module, data_loader = data_loader, inference = True)

    elif config.model.name == "cVAEGenerator":
        model = init_obj(config, "model", models_module, data_loader = data_loader)

        import tensorflow as tf

        sample_waves = tf.random.normal([100, data_loader.in_out_shapes['output_shape']])  # Replace with actual dimensions
        sample_conditions = tf.random.normal([100, data_loader.in_out_shapes['input_shape']])  # Replace with actual dimensions

        # Call the model once to initialize the layers and variables
        model.model([sample_waves, sample_conditions])

        print("Loading model...     ", end="\r")
        model.model.load(os.path.join(model_dir, "best_model.hdf5"))

        return model.model, data_loader
    
    else:
        model = init_obj(config, "model", models_module, data_loader = data_loader)
        
    print("Loading model...     ", end="\r")
    model.load(os.path.join(model_dir, "best_model.hdf5"))

    return model.model, data_loader