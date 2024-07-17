import tensorflow as tf
import argparse
import os
import keras
import numpy as np
import joblib
from utils.config import process_config, init_obj
from utils.data_preprocessing import load_data, get_data_split
from utils.loss import *
import data_loader.gw_dataloader as data_loader_module
import models.gw_models as models_module
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print("\n")

def main():

    parser = argparse.ArgumentParser(description='Script for fine tuning a surrogate trained model.')
    parser.add_argument('-ld', '--load_dir', dest='model_path', help='Checkpoint folder path for model for transfer learning.', metavar='')
    parser.add_argument('-ls', '--loss', dest='loss', help='Whether to use overlap + MAE or only MAE as loss function.', metavar='')
    parser.add_argument('-e', '--epochs', dest='epochs', help='Number of epochs for transfer learning.', default=500, metavar='')
    args = parser.parse_args()

    config, _ = process_config(os.path.join(args.model_path, "config.json"))

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
    else:
        model = init_obj(config, "model", models_module, data_loader = data_loader)

    model.model.load_weights(os.path.join(args.model_path, "best_model.hdf5"))

    print("Model loaded!", end = "\r")
 
    callbacks = []

    # callbacks.append(
    #     TensorBoard(
    #         log_dir=config.callbacks.tensorboard_log_dir,
    #         write_graph=config.callbacks.tensorboard_write_graph,
    #     )
    # )
    callbacks.append(EarlyStopping(monitor = 'val_loss', patience = 30))
    callbacks.append(ReduceLROnPlateau(monitor = 'loss', factor = 0.5, cooldown = 2, patience = 10, verbose = 1, min_lr = 1e-10))       

    pre_tr = model.model.predict(pars_tr, batch_size = 64, verbose = 0)
    pre_ts = model.model.predict(pars_ts, batch_size = 64, verbose = 0)
    pre_val = model.model.predict(pars_val, batch_size = 64, verbose = 0)

    train_mae_prev = np.mean(mean_absolute_error(pre_tr, data_tr))
    test_mae_prev = np.mean(mean_absolute_error(pre_ts, data_ts))
    val_mae_prev = np.mean(mean_absolute_error(pre_val, data_val))

    train_ovlp_prev = np.mean(overlap(tf.convert_to_tensor(pre_tr), tf.convert_to_tensor(data_tr)))
    test_ovlp_prev = np.mean(overlap(tf.convert_to_tensor(pre_ts), tf.convert_to_tensor(data_ts)))
    val_ovlp_prev = np.mean(overlap(tf.convert_to_tensor(pre_val), tf.convert_to_tensor(data_val)))

    
    if args.loss == 'overlap':

        folder_path = os.path.join(args.model_path, "transfer_learning_overlap")

        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        model.model.compile(optimizer = keras.optimizers.Adam(learning_rate = 1e-6), loss = ovlp_mae_loss, metrics = [overlap, 'mean_absolute_error'])
        history_path = os.path.join(folder_path, "history.bin")
        model_path = os.path.join(folder_path, "best_model.hdf5")
        training_txt_path = os.path.join(folder_path, "training_summary.txt")
    else:

        folder_path = os.path.join(args.model_path, "transfer_learning_mae")

        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        model.model.compile(optimizer = keras.optimizers.Adam(learning_rate = 1e-6), loss = 'mae', metrics = [overlap, 'mean_absolute_error'])
        history_path = os.path.join(folder_path, "history.bin")
        model_path = os.path.join(folder_path, "best_model.hdf5")
        training_txt_path = os.path.join(folder_path, "training_summary.txt")

    
    history_retrain = model.model.fit(
        x = pars_tr,
        y = data_tr,
        validation_data = (pars_val, data_val),
        epochs=int(args.epochs),
        verbose=1,
        batch_size=np.floor(len(data_tr)/20).astype(int),
        callbacks=callbacks
    )

    joblib.dump(history_retrain, history_path)
    model.model.save_weights(model_path)

    pre_tr = model.model.predict(pars_tr, batch_size = 64, verbose = 0)
    pre_ts = model.model.predict(pars_ts, batch_size = 64, verbose = 0)
    pre_val = model.model.predict(pars_val, batch_size = 64, verbose = 0)

    train_mae_post = np.mean(mean_absolute_error(pre_tr, data_tr))
    test_mae_post = np.mean(mean_absolute_error(pre_ts, data_ts))
    val_mae_post = np.mean(mean_absolute_error(pre_val, data_val))

    train_ovlp_post = np.mean(overlap(tf.convert_to_tensor(pre_tr), tf.convert_to_tensor(data_tr)))
    test_ovlp_post = np.mean(overlap(tf.convert_to_tensor(pre_ts), tf.convert_to_tensor(data_ts)))
    val_ovlp_post = np.mean(overlap(tf.convert_to_tensor(pre_val), tf.convert_to_tensor(data_val)))

    with open(training_txt_path, "w") as f:
        f.write(f"Train SXS error:\n - MAE:     {train_mae_prev:.4e} --> {train_mae_post:.4e}\n - Overlap: {train_ovlp_prev:.4e} --> {train_ovlp_post:.4e}\n\n")
        f.write(f"Test SXS error:\n - MAE:     {test_mae_prev:.4e} --> {test_mae_post:.4e}\n - Overlap: {test_ovlp_prev:.4e} --> {test_ovlp_post:.4e}\n\n")
        f.write(f"Validation SXS error:\n - MAE:     {val_mae_prev:.4e} --> {val_mae_post:.4e}\n - Overlap: {val_ovlp_prev:.4e} --> {val_ovlp_post:.4e}\n\n")


if __name__ == '__main__':
    main()
