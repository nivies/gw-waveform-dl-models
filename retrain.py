import tensorflow as tf
import argparse
import os
import keras
import numpy as np
import json
import joblib
from utils.config import process_config, init_obj
from utils.data_preprocessing import load_data, get_data_split
from utils.loss import *
from utils.plot_utils import make_plots, PlotOutputsCallback
import data_loader.gw_dataloader as data_loader_module
import models.gw_models as models_module
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from utils.utils import configure_device, query_gpu
import warnings

def custom_format_warning(message, category, filename, lineno, line=None):
    if line is None:
        line = ""
    return f"{category.__name__}: {message} (File: {filename}, Line: {lineno})\n"


warnings.formatwarning = custom_format_warning

print("\n")
configure_device()
print("\n")

def main():

    parser = argparse.ArgumentParser(description='Script for fine tuning a surrogate trained model.')
    parser.add_argument('-ld', '--load_dir', dest='model_path', help='Checkpoint folder path for model for transfer learning.', metavar='')
    parser.add_argument('-ls', '--loss', dest='loss', help='Whether to use overlap + MAE or only MAE as loss function.', metavar='')
    parser.add_argument('-e', '--epochs', dest='epochs', help='Number of epochs for transfer learning.', default=500, metavar='')
    args = parser.parse_args()

    config, config_dict = process_config(os.path.join(args.model_path, "config.json"))

    if config.data_loader.data_output_type == 'amplitude_phase':
        overlap = overlap_amp_phs
        overlap_batched = overlap_amp_phs_batched
        ovlp_mae_loss = ovlp_mae_loss_amp_phs

    elif config.data_loader.data_output_type == 'hphc':
        overlap = overlap_hphc
        overlap_batched = overlap_hphc_batched
        ovlp_mae_loss = ovlp_mae_loss_hphc

    print("Loading data...", end="\r")
    
    data_loader = init_obj(config, "data_loader", data_loader_module)

    print("Loading model...     ", end = "\r")

    if config.model.name == "RegularizedAutoEncoderGenerator":
        model = init_obj(config, "model", models_module, data_loader = data_loader, inference = True)
    else:
        model = init_obj(config, "model", models_module, data_loader = data_loader)

    model.model.load_weights(os.path.join(args.model_path, "best_model.hdf5"))
    
    try:
        for layer_model in model.model.layers:
            for layer in layer_model.layers:
                layer.trainable = True
                if layer.name == 'latent_components':
                    layer.kernel_regularizer = None
    except:
        for layer in model.model.layers:
            layer.trainable = True
            if layer.name == 'latent_components':
                    layer.kernel_regularizer = None


    model = model.model

    print("Model loaded!", end = "\r")
 
    callbacks = []

    # callbacks.append(
    #     TensorBoard(
    #         log_dir=config.callbacks.tensorboard_log_dir,
    #         write_graph=config.callbacks.tensorboard_write_graph,
    #     )
    # )
    callbacks.append(EarlyStopping(monitor = 'val_loss', patience = config.callbacks.early_stopping_patience))
    callbacks.append(ReduceLROnPlateau(monitor = 'loss', factor = config.callbacks.lr_reduce_factor, patience = config.callbacks.lr_reduce_patience, verbose = 1, min_lr = config.callbacks.min_lr))       
    callbacks.append(GradientDiagnosticsCallback(config, data_loader, folder_name = "retrain_gradient_control", batch_size = 32, threshold=1e-6, plot_interval=5, autoencoder = False))
    callbacks.append(PlotOutputsCallback(config, data_loader, folder_name = "retrain_online_figures"))

    with tf.device('/CPU:0'):
        pre_tr = model.predict(data_loader.X_train, batch_size = 256)
        pre_ts = model.predict(data_loader.X_test, batch_size = 256)

        if len(pre_tr) == 2:

            warnings.warn("Multiple output model posed for retraining. Only last output taken into account.", category = UserWarning)

            pre_tr = pre_tr[-1]
            pre_ts = pre_ts[-1]

            model = keras.Model(inputs = model.input, outputs = model.output[1])


        train_mae_prev = np.mean(mean_absolute_error_batched(tf.convert_to_tensor(pre_tr, dtype = tf.float32), tf.convert_to_tensor(data_loader.y_train, dtype = tf.float32), batch_size = 256))
        test_mae_prev = np.mean(mean_absolute_error_batched(tf.convert_to_tensor(pre_ts, dtype = tf.float32), tf.convert_to_tensor(data_loader.y_test, dtype = tf.float32), batch_size = 256))

        train_ovlp_prev = np.mean(overlap_batched(tf.convert_to_tensor(pre_tr, dtype = tf.float32), tf.convert_to_tensor(data_loader.y_train, dtype = tf.float32), batch_size = 256))
        test_ovlp_prev = np.mean(overlap_batched(tf.convert_to_tensor(pre_ts, dtype = tf.float32), tf.convert_to_tensor(data_loader.y_test, dtype = tf.float32), batch_size = 256))


    root_path = os.path.dirname(args.model_path)
    
    if args.loss == 'overlap':

        folder_path = os.path.join(root_path, "retraining_overlap")

        model.summary()

        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        model.compile(optimizer = keras.optimizers.Adam(**config.model.optimizer_kwargs), loss = ovlp_mae_loss, metrics = [overlap, 'mean_absolute_error'])
        history_path = os.path.join(folder_path, "history.bin")
        model_path = os.path.join(folder_path, "best_model.hdf5")
        training_txt_path = os.path.join(folder_path, "training_summary.txt")
    else:

        folder_path = os.path.join(root_path, "retraining_mae")

        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        model.compile(optimizer = keras.optimizers.Adam(**config.model.optimizer_kwargs), loss = 'mae', metrics = [overlap, 'mean_absolute_error'])
        history_path = os.path.join(folder_path, "history.bin")
        model_path = os.path.join(folder_path, "best_model.hdf5")
        training_txt_path = os.path.join(folder_path, "training_summary.txt")
    
    with open(os.path.join(folder_path, "config.json"), 'w') as fp:
        json.dump(config_dict, fp)

    try:
        history_retrain = model.fit(
            data_loader.get_train_data(),
            validation_data = data_loader.get_test_data(),
            epochs=int(args.epochs),
            verbose=1,
            batch_size=config.trainer.batch_size,
            callbacks=callbacks
        )
    except:
        history_retrain = model.fit(
            data_loader.get_generator_train_data(),
            validation_data = data_loader.get_generator_test_data(),
            epochs=int(args.epochs),
            verbose=1,
            batch_size=config.trainer.batch_size,
            callbacks=callbacks
        )

    joblib.dump(history_retrain.history, history_path)
    model.save_weights(model_path)

    with tf.device('/CPU:0'):

        pre_tr = model.predict(data_loader.X_train, batch_size = 64, verbose = 0)
        pre_ts = model.predict(data_loader.X_test, batch_size = 64, verbose = 0)

        train_mae_post = np.mean(mean_absolute_error_batched(tf.convert_to_tensor(pre_tr, dtype = tf.float32), tf.convert_to_tensor(data_loader.y_train, dtype = tf.float32), batch_size = 256))
        test_mae_post = np.mean(mean_absolute_error_batched(tf.convert_to_tensor(pre_ts, dtype = tf.float32), tf.convert_to_tensor(data_loader.y_test, dtype = tf.float32), batch_size = 256))

        train_ovlp_post = np.mean(overlap_batched(tf.convert_to_tensor(pre_tr, dtype = tf.float32), tf.convert_to_tensor(data_loader.y_train, dtype = tf.float32), batch_size = 256))
        test_ovlp_post = np.mean(overlap_batched(tf.convert_to_tensor(pre_ts, dtype = tf.float32), tf.convert_to_tensor(data_loader.y_test, dtype = tf.float32), batch_size = 256))

        with open(training_txt_path, "w") as f:
            f.write(f"Train error:\n - MAE:     {train_mae_prev:.4e} --> {train_mae_post:.4e}\n - Overlap: {train_ovlp_prev:.4e} --> {train_ovlp_post:.4e}\n\n")
            f.write(f"Test error:\n - MAE:     {test_mae_prev:.4e} --> {test_mae_post:.4e}\n - Overlap: {test_ovlp_prev:.4e} --> {test_ovlp_post:.4e}\n\n")

        config.trainer.uninitialised = True
        make_plots(model = model, dir = os.path.join(folder_path, "retrain_figures"), data_loader = data_loader, config = config, metric = 'overlap')


if __name__ == '__main__':
    main()
