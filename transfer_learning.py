import tensorflow as tf
import argparse
import os
import keras
import numpy as np
import joblib
from utils.config import process_config, init_obj
from utils.data_preprocessing import load_data, get_data_split
from utils.loss import *
from utils.plot_utils import make_plots_sxs
import data_loader.gw_dataloader as data_loader_module
import models.gw_models as models_module
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from utils.utils import configure_device

print("\n")
configure_device()
print("\n")

def main():

    parser = argparse.ArgumentParser(description='Script for fine tuning a surrogate trained model.')
    parser.add_argument('-ld', '--load_dir', dest='model_path', help='Checkpoint folder path for model for transfer learning.', metavar='')
    parser.add_argument('-ls', '--loss', dest='loss', help='Whether to use overlap + MAE or only MAE as loss function.', metavar='')
    parser.add_argument('-e', '--epochs', dest='epochs', help='Number of epochs for transfer learning.', default=500, metavar='')
    args = parser.parse_args()

    config, _ = process_config(os.path.join(args.model_path, "config.json"))

    if config.data_loader.data_output_type == 'amplitude_phase':
        overlap = overlap_amp_phs
        overlap_batched = overlap_amp_phs_batched
        ovlp_mae_loss = ovlp_mae_loss_amp_phs

    elif config.data_loader.data_output_type == 'hphc':
        overlap = overlap_hphc
        overlap_batched = overlap_hphc_batched
        ovlp_mae_loss = ovlp_mae_loss_hphc

    print("Loading data...", end="\r")

    if "_q_" in config.data_loader.data_path:
        sxs_pars, sxs_data, _ = load_data("./data/sxs_q_data.hdf5", config.data_loader.data_output_type)

    elif "_qz_" in config.data_loader.data_path:
        sxs_pars, sxs_data, _ = load_data("./data/sxs_qz_data.hdf5", config.data_loader.data_output_type)
        
    elif "_qzp_" in config.data_loader.data_path:
        sxs_pars, sxs_data, _ = load_data("./data/sxs_qzp_data.hdf5", config.data_loader.data_output_type)

    
    data_tr, data_ts, data_val, pars_tr, pars_ts, pars_val = get_data_split(sxs_data, sxs_pars, split = [0.4, 0.4, 0.2])
    data_loader = init_obj(config, "data_loader", data_loader_module)

    print("Loading model...     ", end = "\r")

    if config.model.name == "RegularizedAutoEncoderGenerator":
        model = init_obj(config, "model", models_module, data_loader = data_loader, inference = True)

    elif config.model.name == "cVAEGenerator":
        model = init_obj(config, "model", models_module, data_loader = data_loader)

        sample_waves = tf.random.normal([100, data_loader.in_out_shapes['output_shape']])  # Replace with actual dimensions
        sample_conditions = tf.random.normal([100, data_loader.in_out_shapes['input_shape']])  # Replace with actual dimensions

        # Call the model once to initialize the layers and variables
        model.model([sample_waves, sample_conditions])
    
    else:
        model = init_obj(config, "model", models_module, data_loader = data_loader)

    model.model.load_weights(os.path.join(args.model_path, "best_model.hdf5"))

    # try:
    #     for layer_model in model.model.layers:
    #         for layer in layer_model.layers:
    #             layer.trainable = True
    #             if layer.name == 'latent_components':
    #                 layer.kernel_regularizer = None
    # except:
    #     for layer in model.model.layers:
    #         layer.trainable = True
    #         if layer.name == 'latent_components':
    #             layer.kernel_regularizer = None
    # model = model.model

    for layer in model.model.layers:
        layer.trainable = True
        if layer.name == 'latent_components':
            layer.kernel_regularizer = None

    inp = model.model.input
    x = model.model.layers[0](inp)
    for layer in model.model.layers[1:-1]:
        x = layer(x)
    opt = model.model.layers[-1](x)

    model = keras.Model(inp, opt)

    print("Model loaded!", end = "\r")
 
    callbacks = []

    # callbacks.append(
    #     TensorBoard(
    #         log_dir=config.callbacks.tensorboard_log_dir,
    #         write_graph=config.callbacks.tensorboard_write_graph,
    #     )
    # )
    callbacks.append(EarlyStopping(monitor = 'val_loss', patience = 500))
    callbacks.append(ReduceLROnPlateau(monitor = 'loss', factor = 0.5, cooldown = 2, patience = 30, verbose = 1, min_lr = 1e-10))       


    if config.model.name == "cVAEGenerator":

        rand_tr = tf.random.normal(shape = (pars_tr.shape[0], config.model.latent_dim))
        rand_ts = tf.random.normal(shape = (pars_ts.shape[0], config.model.latent_dim))
        rand_val = tf.random.normal(shape = (pars_val.shape[0], config.model.latent_dim))

        pre_tr = model.decoder.predict([rand_tr, pars_tr], batch_size = 1024)
        pre_ts = model.decoder.predict([rand_ts, pars_ts], batch_size = 1024)
        pre_val = model.decoder.predict([rand_val, pars_val], batch_size = 1024)    

    else:
        pre_tr = model.predict(pars_tr, batch_size = 1024)
        pre_ts = model.predict(pars_ts, batch_size = 1024)
        pre_val = model.predict(pars_val, batch_size = 1024)

    train_mae_prev = np.mean(mean_absolute_error_batched(tf.convert_to_tensor(pre_tr, dtype = tf.float32), tf.convert_to_tensor(data_tr, dtype = tf.float32), batch_size = 64))
    test_mae_prev = np.mean(mean_absolute_error_batched(tf.convert_to_tensor(pre_ts, dtype = tf.float32), tf.convert_to_tensor(data_ts, dtype = tf.float32), batch_size = 64))
    val_mae_prev = np.mean(mean_absolute_error_batched(tf.convert_to_tensor(pre_val, dtype = tf.float32), tf.convert_to_tensor(data_val, dtype = tf.float32), batch_size = 64))

    train_ovlp_prev = np.mean(overlap_batched(tf.convert_to_tensor(pre_tr), tf.convert_to_tensor(data_tr), batch_size = 64))
    test_ovlp_prev = np.mean(overlap_batched(tf.convert_to_tensor(pre_ts), tf.convert_to_tensor(data_ts), batch_size = 64))
    val_ovlp_prev = np.mean(overlap_batched(tf.convert_to_tensor(pre_val), tf.convert_to_tensor(data_val), batch_size = 64))

    root_path = os.path.dirname(args.model_path)
    
    if args.loss == 'overlap':

        folder_path = os.path.join(root_path, "transfer_learning_overlap")

        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        model.compile(optimizer = keras.optimizers.Adam(learning_rate = config.model.optimizer_kwargs.learning_rate/100), loss = ovlp_mae_loss, metrics = [overlap, 'mean_absolute_error'])
        history_path = os.path.join(folder_path, "history.bin")
        model_path = os.path.join(folder_path, "best_model.hdf5")
        training_txt_path = os.path.join(folder_path, "training_summary.txt")
    else:

        folder_path = os.path.join(root_path, "transfer_learning_mae")

        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        model.compile(optimizer = keras.optimizers.Adam(learning_rate = config.model.optimizer_kwargs.learning_rate/100), loss = 'mae', metrics = [overlap, 'mean_absolute_error'])
        history_path = os.path.join(folder_path, "history.bin")
        model_path = os.path.join(folder_path, "best_model.hdf5")
        training_txt_path = os.path.join(folder_path, "training_summary.txt")

    if config.model.name == "cVAEGenerator":

        # model.encoder.trainable = False

        history_retrain = model.fit(
            x = [data_tr, pars_tr],
            y = data_tr,
            validation_data = ([data_val, pars_val], data_val),
            epochs=int(args.epochs),
            verbose=1,
            batch_size=np.floor(len(data_tr)/20).astype(int),
            callbacks=callbacks
        )

    else:
        history_retrain = model.fit(
            x = pars_tr,
            y = data_tr,
            validation_data = (pars_val, data_val),
            epochs=int(args.epochs),
            verbose=1,
            batch_size=np.floor(len(data_tr)/20).astype(int),
            callbacks=callbacks
        )

    joblib.dump(history_retrain, history_path)
    model.save_weights(model_path)

    if config.model.name == "cVAEGenerator":

        rand_tr = tf.random.normal(shape = (pars_tr.shape[0], config.model.latent_dim))
        rand_ts = tf.random.normal(shape = (pars_ts.shape[0], config.model.latent_dim))
        rand_val = tf.random.normal(shape = (pars_val.shape[0], config.model.latent_dim))

        pre_tr = model.decoder.predict([rand_tr, pars_tr], batch_size = 1024)
        pre_ts = model.decoder.predict([rand_ts, pars_ts], batch_size = 1024)
        pre_val = model.decoder.predict([rand_val, pars_val], batch_size = 1024)    

    else:
        pre_tr = model.predict(pars_tr, batch_size = 1024)
        pre_ts = model.predict(pars_ts, batch_size = 1024)
        pre_val = model.predict(pars_val, batch_size = 1024)

    train_mae_post = np.mean(mean_absolute_error_batched(tf.convert_to_tensor(pre_tr, dtype = tf.float32), tf.convert_to_tensor(data_tr, dtype = tf.float32), batch_size = 64))
    test_mae_post = np.mean(mean_absolute_error_batched(tf.convert_to_tensor(pre_ts, dtype = tf.float32), tf.convert_to_tensor(data_ts, dtype = tf.float32), batch_size = 64))
    val_mae_post = np.mean(mean_absolute_error_batched(tf.convert_to_tensor(pre_val, dtype = tf.float32), tf.convert_to_tensor(data_val, dtype = tf.float32), batch_size = 64))

    train_ovlp_post = np.mean(overlap_batched(tf.convert_to_tensor(pre_tr), tf.convert_to_tensor(data_tr), batch_size = 64))
    test_ovlp_post = np.mean(overlap_batched(tf.convert_to_tensor(pre_ts), tf.convert_to_tensor(data_ts), batch_size = 64))
    val_ovlp_post = np.mean(overlap_batched(tf.convert_to_tensor(pre_val), tf.convert_to_tensor(data_val), batch_size = 64))

    with open(training_txt_path, "w") as f:
        f.write(f"Train SXS error:\n - MAE:     {train_mae_prev:.4e} --> {train_mae_post:.4e}\n - Overlap: {train_ovlp_prev:.4e} --> {train_ovlp_post:.4e}\n\n")
        f.write(f"Test SXS error:\n - MAE:     {test_mae_prev:.4e} --> {test_mae_post:.4e}\n - Overlap: {test_ovlp_prev:.4e} --> {test_ovlp_post:.4e}\n\n")
        f.write(f"Validation SXS error:\n - MAE:     {val_mae_prev:.4e} --> {val_mae_post:.4e}\n - Overlap: {val_ovlp_prev:.4e} --> {val_ovlp_post:.4e}\n\n")

    make_plots_sxs(model = model, dir = os.path.join(folder_path, "tl_figures"), config = config, metric = 'overlap')


if __name__ == '__main__':
    main()
