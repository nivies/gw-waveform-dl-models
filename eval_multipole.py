import tensorflow as tf
from utils.eval import load_model_and_data_loader
from utils.config import process_config
from utils.plot_utils import *
from data_loader.gw_dataloader import GWDataLoader
import keras

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model_dir = "/home/nino/GW/Keras-Project-Template/experiments/multipole_q_reg/full_model.hdf5"
config_path = "/home/nino/GW/Keras-Project-Template/experiments/q_bbh_reg_light/reg_q_bbh_light/checkpoints/config.json"
data_folder = "/home/nino/GW/Keras-Project-Template/data/q_multipole"

config, _ = process_config(config_path)
model, data_loader = load_model_and_data_loader("/home/nino/GW/Keras-Project-Template/experiments/q_bbh_reg_light/reg_q_bbh_light/checkpoints")

def define_model(base_model):

    nozz_22 = keras.models.clone_model(keras.Model(base_model.layers[-1].layers[-1].input, base_model.layers[-1].layers[-1].output))
    nozz_21 = keras.models.clone_model(keras.Model(base_model.layers[-1].layers[-2].input, base_model.layers[-1].layers[-1].output))
    nozz_20 = keras.models.clone_model(keras.Model(base_model.layers[-1].layers[-3].input, base_model.layers[-1].layers[-1].output))
    nozz_2n1 = keras.models.clone_model(keras.Model(base_model.layers[-1].layers[-2].input, base_model.layers[-1].layers[-1].output))
    nozz_2n2 = keras.models.clone_model(keras.Model(base_model.layers[-1].layers[-1].input, base_model.layers[-1].layers[-1].output))

    inp = base_model.layers[1].input

    x = base_model.layers[1](inp)
    x = base_model.layers[2](x)

    for layer in base_model.layers[3].layers[:-4]:
        x = layer(x)

    mp20 = base_model.layers[3].layers[-4](x)
    mp21 = base_model.layers[3].layers[-3](mp20)
    mp22 = base_model.layers[3].layers[-2](mp21)

    opt_22 = nozz_22(mp22)
    opt_2n2 = nozz_2n2(mp22)
    opt_21 = nozz_21(mp21)
    opt_2n1 = nozz_2n1(mp21)
    opt_20 = nozz_20(mp20)
    
    return keras.Model(inp, [opt_20, opt_21, opt_2n1, opt_22, opt_2n2])

def make_plots(dir, data_loader, y_pred_tr, y_pred_ts):

    dir_tr, dir_ts = make_plotting_dirs(dir)

    mm_tr, bst_tr, wrst_tr = pycbc_mismatch(y_pred_tr, data_loader.y_train, data_loader.delta_t)
    mm_ts, bst_ts, wrst_ts = pycbc_mismatch(y_pred_ts, data_loader.y_test, data_loader.delta_t)

    perc_10_tr = get_percentile_index(mm_tr, 0.9)
    perc_50_tr = get_percentile_index(mm_tr, 0.5)

    perc_10_ts = get_percentile_index(mm_ts, 0.9)
    perc_50_ts = get_percentile_index(mm_ts, 0.5)

    dump(mm_tr, os.path.join(dir, "mismatches_train.bin"))
    dump(mm_ts, os.path.join(dir, "mismatches_test.bin"))

    print("Plotting histogram", end='\r')
    plot_mismatch_histogram(mm_tr, dir_tr, True)
    plot_mismatch_histogram(mm_ts, dir_ts, False)

    print("Plotting waveforms", end='\r')
    plot_waveform_comparison(y_pred_tr[bst_tr], data_loader.y_train[bst_tr], data_loader.delta_t, dir_tr, title = f"Best train case. Mismatch: {mm_tr[bst_tr]:.3e}", save_name = "best_waveform.jpg")
    plot_waveform_comparison(y_pred_ts[bst_ts], data_loader.y_test[bst_ts], data_loader.delta_t, dir_ts, title = f"Best test case. Mismatch: {mm_ts[bst_ts]:.3e}", save_name = "best_waveform.jpg")
    
    plot_waveform_comparison(y_pred_tr[wrst_tr], data_loader.y_train[wrst_tr], data_loader.delta_t, dir_tr, title = f"Worst train case. Mismatch: {mm_tr[wrst_tr]:.3e}", save_name = "worst_waveform.jpg")
    plot_waveform_comparison(y_pred_ts[wrst_ts], data_loader.y_test[wrst_ts], data_loader.delta_t, dir_ts, title = f"Worst test case. Mismatch: {mm_ts[wrst_ts]:.3e}", save_name = "worst_waveform.jpg")

    plot_waveform_comparison(y_pred_tr[perc_10_tr], data_loader.y_train[perc_10_tr], data_loader.delta_t, dir_tr, title = f"10th percentile train waveform. Mismatch: {mm_tr[perc_10_tr]:.3e}", save_name = "perc_10_waveform.jpg")
    plot_waveform_comparison(y_pred_ts[perc_10_ts], data_loader.y_test[perc_10_ts], data_loader.delta_t, dir_ts, title = f"10th percentile test waveform. Mismatch: {mm_ts[perc_10_ts]:.3e}", save_name = "perc_10_waveform.jpg")

    plot_waveform_comparison(y_pred_tr[perc_50_tr], data_loader.y_train[perc_50_tr], data_loader.delta_t, dir_tr, title = f"50th percentile train waveform. Mismatch: {mm_tr[perc_50_tr]:.3e}", save_name = "perc_50_waveform.jpg")
    plot_waveform_comparison(y_pred_ts[perc_50_ts], data_loader.y_test[perc_50_ts], data_loader.delta_t, dir_ts, title = f"50th percentile test waveform. Mismatch: {mm_ts[perc_50_ts]:.3e}", save_name = "perc_50_waveform.jpg")

multipole_model = define_model(model)
multipole_model.load_weights(model_dir)

y_pred_tr = multipole_model.predict(data_loader.X_train, batch_size = 2048)
y_pred_ts = multipole_model.predict(data_loader.X_test, batch_size = 2048)

figure_path = "/home/nino/GW/Keras-Project-Template/figures/q_bbh_multipoles"

for multipole_data in os.listdir(data_folder):
    print(multipole_data + "\n")
    config.data_loader.data_path = os.path.join(data_folder, multipole_data)
    data_loader = GWDataLoader(config)
    
    if "20" in multipole_data:
        y_prd_tr = y_pred_tr[0]
        y_prd_ts = y_pred_ts[0]
        folder = "mode_20/"

    elif "21" in multipole_data:
        y_prd_tr = y_pred_tr[1]
        y_prd_ts = y_pred_ts[1]
        folder = "mode_21/"

    elif "2n1" in multipole_data:
        y_prd_tr = y_pred_tr[2]
        y_prd_ts = y_pred_ts[2]
        folder = "mode_2n1/"

    elif "22" in multipole_data:
        y_prd_tr = y_pred_tr[3]
        y_prd_ts = y_pred_ts[3]
        folder = "mode_22/"

    elif "2n2" in multipole_data:
        y_prd_tr = y_pred_tr[4]
        y_prd_ts = y_pred_ts[4]
        folder = "mode_2n2/"

    make_plots(os.path.join(figure_path, folder), data_loader, y_prd_tr, y_prd_ts)