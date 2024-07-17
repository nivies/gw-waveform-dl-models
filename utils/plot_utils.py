import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
from utils.data_preprocessing import load_data, get_data_split
from joblib import dump
from pycbc.filter.matchedfilter import match
from pycbc.types import TimeSeries
from tqdm import tqdm
import tensorflow as tf
import keras.backend as K

def wavewise_overlap(h1, h2, config, dt=2*4.925794970773135e-06, df=None):
    
    split_size = int(4096/2)

    if config.data_loader.data_output_type == 'amplitude_phase':

        h1_amp, h1_phs = h1[:split_size], h1[split_size:]
        h2_amp, h2_phs = h2[:split_size], h2[split_size:]

        h1 =  tf.cast(h1_amp, tf.complex64)*tf.math.exp(1j*(tf.cast(h1_phs, tf.complex64)))
        h2 = tf.cast(h2_amp, tf.complex64)*tf.math.exp(1j*(tf.cast(h2_phs, tf.complex64)))

    elif config.data_loader.data_output_type == 'hphc':

        h1_hp, h1_hc = h1[:split_size], h1[split_size:]
        h2_hp, h2_hc = h2[:split_size], h2[split_size:]

        h1 =  tf.cast(h1_hp, tf.complex64) + 1j*(tf.cast(h1_hc, tf.complex64))
        h2 = tf.cast(h2_hp, tf.complex64) + 1j*(tf.cast(h2_hc, tf.complex64))

    h1_f = tf.signal.fft(h1)*dt
    h2_f = tf.signal.fft(h2)*dt
    
    df = 1.0 /  2048 / dt
    sig_norm = 4*df

    sig1 = K.sqrt(tf.cast((tf.math.reduce_sum(tf.math.conj(h1_f)*h1_f,axis=-1)),tf.float32)*sig_norm)
    sig2 = K.sqrt(tf.cast((tf.math.reduce_sum(tf.math.conj(h2_f)*h2_f,axis=-1)),tf.float32)*sig_norm)
    
    norm = 1/sig1/sig2
    inner = tf.cast(tf.math.reduce_sum((tf.math.conj(h1_f)*h2_f),axis=-1),tf.float32)
    overl = tf.cast((4*df*inner*norm),tf.float32)
    
    return  K.abs(1. - overl)

def osvaldos_overlap(y_pred, y_true, delta_t, config):

    N = y_pred.shape[0]
    mm = []

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.convert_to_tensor(y_true)

    for y_prd, y_tr in tqdm(zip(y_pred, y_true), total = N, desc = "Calculating 1 - overlap"):

        mm.append(wavewise_overlap(h1 = y_prd, h2 = y_tr, dt = delta_t, config = config))

    return np.array(mm), np.argmin(mm), np.argmax(mm)

def get_percentile_index(mm, percentile):
    sorted_array = np.sort(mm)
    perc_sorted_idx = int(percentile * len(np.sort(mm)))
    value_perc_10_tr = sorted_array[perc_sorted_idx]
    perc_idx = np.where(mm == value_perc_10_tr)[0][0]

    return perc_idx

def make_plotting_dirs(dir, val = False):
    
    if not os.path.isdir(dir):
        os.mkdir(dir)

    if val:

        dir_train = os.path.join(dir, "train/")
        dir_test = os.path.join(dir, "test/")
        dir_val = os.path.join(dir, "validation/")

        if not os.path.isdir(dir_train):
            os.mkdir(dir_train)
        if not os.path.isdir(dir_test):
            os.mkdir(dir_test)
        if not os.path.isdir(dir_val):
            os.mkdir(dir_val)
        
        return dir_train, dir_test, dir_val
    
    else:
        dir_train = os.path.join(dir, "train/")
        dir_test = os.path.join(dir, "test/")

        if not os.path.isdir(dir_train):
            os.mkdir(dir_train)
        if not os.path.isdir(dir_test):
            os.mkdir(dir_test)
        
        return dir_train, dir_test
    
def pycbc_mismatch(y_pred, y_true, delta_t, real = "real"):

    '''
    delta_t must be in seconds
    '''

    mm = []
    for prd, grnd in tqdm(zip(y_pred, y_true), total = len(y_pred), desc = "Calculating mismatches"):
        if real == "real":
            pred_ts = TimeSeries(np.real(prd).astype(float), delta_t)
            grnd_ts = TimeSeries(np.real(grnd).astype(float), delta_t)
        elif real == "imag":
            pred_ts = TimeSeries(np.imag(prd).astype(float), delta_t)
            grnd_ts = TimeSeries(np.imag(grnd).astype(float), delta_t)
        mm.append(1-match(pred_ts, grnd_ts)[0])
    
    return np.array(mm), np.argmin(mm), np.argmax(mm)

def plot_mismatch_histogram(mismatches, dir, train_test):

    matplotlib.rc('font', size=16)
    plt.rcParams["figure.figsize"] = (10,7)

    bins = np.logspace(np.log10(min(mismatches)), np.log10(max(mismatches)), num=30)
    _ = plt.hist(mismatches, bins = bins)
    plt.grid(True, which="both", axis="both", linestyle='-', color='gray', linewidth=0.5, alpha = 0.7)
    plt.xscale('log')
    plt.title(f'{train_test} mean mismatch: {np.mean(mismatches):3e}')
    plt.ylabel('Count')
    plt.xlabel(f'{train_test} mismatch')
    plt.savefig(os.path.join(dir, "mismatch_histogram.png"))
    plt.clf()

def plot_waveform_comparison(y_pred, y_true, delta_t, dir, title, save_name):

    matplotlib.rc('font', size=16)
    plt.rcParams["figure.figsize"] = (12,8)

    total_length = len(y_pred)
    times = np.arange(total_length) * delta_t

    fig, ax = plt.subplots()

    max_amp = max([np.max(np.abs(y_true)), np.max(np.abs(y_pred))])

    ax.plot(times, y_pred, 'r', linewidth = 1.5, label = "Generated waveform")
    ax.plot(times, y_true, 'k--', linewidth = 1, label = "Targeted waveform")
    plt.legend()
    plt.title(title)
    
    plt.ylabel("Strain")
    plt.xlabel("Time (s)")

    # inset axes....
    if total_length == 2048:
        x1, x2, y1, y2 = times[1850], times[-1], -max_amp*1.1, max_amp*1.1
    else:
        x1, x2, y1, y2 = times[3200], times[3800], -max_amp*1.1, max_amp*1.1

    axins = ax.inset_axes(
        [0.06, 0.73, 0.73, 0.25],
        xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    axins.plot(times, y_pred, 'r', linewidth = 1.5)
    axins.plot(times, y_true, 'k--', linewidth = 1)

    ax.indicate_inset_zoom(axins, edgecolor="black")

    fig.savefig(os.path.join(dir, save_name))
    plt.close()
    return

def plot_mismatch_comparison(dir, data_dict, train_test):

    matplotlib.rc('font', size=17)
    plt.rcParams["figure.figsize"] = (9,6)

    if not os.path.isdir(dir):
        os.mkdir(dir)

    data = [data_dict[key] for key in data_dict.keys()]
    bins = np.logspace(np.log10(min([min(vec) for vec in data])), np.log10(max([max(vec) for vec in data])), num=30)

    for key in data_dict.keys():

        _ = plt.hist(data_dict[key], bins = bins, histtype='step', label = f"{key}.  $\\mu$ = {np.mean(data_dict[key]):.2e}")

    plt.legend()
    plt.grid(True, which="both", axis="both", linestyle='-', color='gray', linewidth=0.5, alpha = 0.7)
    plt.xscale('log')
    plt.title(f'{train_test} set mismatch comparisons')
    plt.ylabel('Count')
    plt.xlabel('Mismatch')
    plt.savefig(os.path.join(dir, "mismatch_comparison.jpg"))
    plt.close()

def make_plots(model, dir, data_loader, config, metric):

    dir_tr, dir_ts = make_plotting_dirs(dir)

    y_pred_tr = model.predict(data_loader.X_train, batch_size = 1024)
    y_pred_ts = model.predict(data_loader.X_test, batch_size = 1024)

    if metric == "overlap":

        split_idx = int(y_pred_tr.shape[-1]/2)
        mm_tr, bst_tr, wrst_tr = osvaldos_overlap(y_pred_tr, data_loader.y_train, data_loader.delta_t, config)
        mm_ts, bst_ts, wrst_ts = osvaldos_overlap(y_pred_ts, data_loader.y_test, data_loader.delta_t, config)
        
        y_pred_tr = np.real(y_pred_tr[:, :split_idx]*np.exp(1.0j*y_pred_tr[:, split_idx:]))
        y_pred_ts = np.real(y_pred_ts[:, :split_idx]*np.exp(1.0j*y_pred_ts[:, split_idx:]))

        data_loader.y_train = np.real(data_loader.y_train[:, :split_idx]*np.exp(1.0j*data_loader.y_train[:, split_idx:]))
        data_loader.y_test = np.real(data_loader.y_test[:, :split_idx]*np.exp(1.0j*data_loader.y_test[:, split_idx:]))

    elif config.data_loader.data_output_type == "amplitude_phase":

        split_idx = int(y_pred_tr.shape[-1]/2)
        y_pred_tr = y_pred_tr[:, :split_idx]*np.exp(1.0j*y_pred_tr[:, split_idx:])
        y_pred_ts = y_pred_ts[:, :split_idx]*np.exp(1.0j*y_pred_ts[:, split_idx:])
        
        data_loader.y_train = np.real(data_loader.y_train[:, :split_idx]*np.exp(1.0j*data_loader.y_train[:, split_idx:]))
        data_loader.y_test = np.real(data_loader.y_test[:, :split_idx]*np.exp(1.0j*data_loader.y_test[:, split_idx:]))

        mm_tr, bst_tr, wrst_tr = pycbc_mismatch(y_pred_tr, data_loader.y_train, data_loader.delta_t, metric)
        mm_ts, bst_ts, wrst_ts = pycbc_mismatch(y_pred_ts, data_loader.y_test, data_loader.delta_t, metric)

        y_pred_tr = np.real(y_pred_tr)
        y_pred_ts = np.real(y_pred_ts)

    
    else:

        mm_tr, bst_tr, wrst_tr = pycbc_mismatch(y_pred_tr, data_loader.y_train, data_loader.delta_t, metric)
        mm_ts, bst_ts, wrst_ts = pycbc_mismatch(y_pred_ts, data_loader.y_test, data_loader.delta_t, metric)

    perc_10_tr = get_percentile_index(mm_tr, 0.9)
    perc_50_tr = get_percentile_index(mm_tr, 0.5)

    perc_10_ts = get_percentile_index(mm_ts, 0.9)
    perc_50_ts = get_percentile_index(mm_ts, 0.5)

    dump(mm_tr, os.path.join(dir, "mismatches_train.bin"))
    dump(mm_ts, os.path.join(dir, "mismatches_test.bin"))

    print("Plotting histogram", end='\r')
    plot_mismatch_histogram(mm_tr, dir_tr, "Train")
    plot_mismatch_histogram(mm_ts, dir_ts, "Test")

    print("Plotting waveforms", end='\r')
    plot_waveform_comparison(y_pred_tr[bst_tr], data_loader.y_train[bst_tr], data_loader.delta_t, dir_tr, title = f"Best train case. Mismatch: {mm_tr[bst_tr]:.3e}", save_name = "best_waveform.jpg")
    plot_waveform_comparison(y_pred_ts[bst_ts], data_loader.y_test[bst_ts], data_loader.delta_t, dir_ts, title = f"Best test case. Mismatch: {mm_ts[bst_ts]:.3e}", save_name = "best_waveform.jpg")
    
    plot_waveform_comparison(y_pred_tr[wrst_tr], data_loader.y_train[wrst_tr], data_loader.delta_t, dir_tr, title = f"Worst train case. Mismatch: {mm_tr[wrst_tr]:.3e}", save_name = "worst_waveform.jpg")
    plot_waveform_comparison(y_pred_ts[wrst_ts], data_loader.y_test[wrst_ts], data_loader.delta_t, dir_ts, title = f"Worst test case. Mismatch: {mm_ts[wrst_ts]:.3e}", save_name = "worst_waveform.jpg")

    plot_waveform_comparison(y_pred_tr[perc_10_tr], data_loader.y_train[perc_10_tr], data_loader.delta_t, dir_tr, title = f"10th percentile train waveform. Mismatch: {mm_tr[perc_10_tr]:.3e}", save_name = "perc_10_waveform.jpg")
    plot_waveform_comparison(y_pred_ts[perc_10_ts], data_loader.y_test[perc_10_ts], data_loader.delta_t, dir_ts, title = f"10th percentile test waveform. Mismatch: {mm_ts[perc_10_ts]:.3e}", save_name = "perc_10_waveform.jpg")

    plot_waveform_comparison(y_pred_tr[perc_50_tr], data_loader.y_train[perc_50_tr], data_loader.delta_t, dir_tr, title = f"50th percentile train waveform. Mismatch: {mm_tr[perc_50_tr]:.3e}", save_name = "perc_50_waveform.jpg")
    plot_waveform_comparison(y_pred_ts[perc_50_ts], data_loader.y_test[perc_50_ts], data_loader.delta_t, dir_ts, title = f"50th percentile test waveform. Mismatch: {mm_ts[perc_50_ts]:.3e}", save_name = "perc_50_waveform.jpg")

def make_plots_sxs(model, dir, config, metric):

    if "_q_" in config.data_loader.data_path:
        sxs_pars, sxs_data, delta_t = load_data("./data/sxs_q_data.hdf5", config.data_loader.data_output_type)

    elif "_qz_" in config.data_loader.data_path:
        sxs_pars, sxs_data, delta_t = load_data("./data/sxs_qz_data.hdf5", config.data_loader.data_output_type)
        
    elif "_qzp_" in config.data_loader.data_path:
        sxs_pars, sxs_data, delta_t = load_data("./data/sxs_qzp_data.hdf5", config.data_loader.data_output_type)

    
    data_tr, data_ts, data_val, pars_tr, pars_ts, pars_val = get_data_split(sxs_data, sxs_pars, split = [0.4, 0.4, 0.2])

    dir_tr, dir_ts, dir_val = make_plotting_dirs(dir, val=True)

    y_pred_tr = model.predict(pars_tr, batch_size = 1024)
    y_pred_ts = model.predict(pars_ts, batch_size = 1024)
    y_pred_val = model.predict(pars_val, batch_size = 1024)

    if metric == "overlap":

        split_idx = int(y_pred_tr.shape[-1]/2)
        mm_tr, bst_tr, wrst_tr = osvaldos_overlap(y_pred_tr, data_tr, delta_t)
        mm_ts, bst_ts, wrst_ts = osvaldos_overlap(y_pred_ts, data_ts, delta_t)
        mm_val, bst_val, wrst_val = osvaldos_overlap(y_pred_val, data_val, delta_t)
        
        y_pred_tr = np.real(y_pred_tr[:, :split_idx]*np.exp(1.0j*y_pred_tr[:, split_idx:]))
        y_pred_ts = np.real(y_pred_ts[:, :split_idx]*np.exp(1.0j*y_pred_ts[:, split_idx:]))
        y_pred_val = np.real(y_pred_val[:, :split_idx]*np.exp(1.0j*y_pred_val[:, split_idx:]))

        data_tr = np.real(data_tr[:, :split_idx]*np.exp(1.0j*data_tr[:, split_idx:]))
        data_ts = np.real(data_ts[:, :split_idx]*np.exp(1.0j*data_ts[:, split_idx:]))
        data_val = np.real(data_val[:, :split_idx]*np.exp(1.0j*data_val[:, split_idx:]))

    elif config.data_loader.data_output_type == "amplitude_phase":

        split_idx = int(y_pred_tr.shape[-1]/2)
        y_pred_tr = y_pred_tr[:, :split_idx]*np.exp(1.0j*y_pred_tr[:, split_idx:])
        y_pred_ts = y_pred_ts[:, :split_idx]*np.exp(1.0j*y_pred_ts[:, split_idx:])
        y_pred_val = y_pred_val[:, :split_idx]*np.exp(1.0j*y_pred_val[:, split_idx:])
        
        data_tr = np.real(data_tr[:, :split_idx]*np.exp(1.0j*data_tr[:, split_idx:]))
        data_ts = np.real(data_ts[:, :split_idx]*np.exp(1.0j*data_ts[:, split_idx:]))
        data_val = np.real(data_val[:, :split_idx]*np.exp(1.0j*data_val[:, split_idx:]))

        mm_tr, bst_tr, wrst_tr = pycbc_mismatch(y_pred_tr, data_tr, delta_t, metric)
        mm_ts, bst_ts, wrst_ts = pycbc_mismatch(y_pred_ts, data_ts, delta_t, metric)
        mm_val, bst_val, wrst_val = pycbc_mismatch(y_pred_val, data_val, delta_t, metric)

        y_pred_tr = np.real(y_pred_tr)
        y_pred_ts = np.real(y_pred_ts)
        y_pred_val = np.real(y_pred_val)

    
    else:

        mm_tr, bst_tr, wrst_tr = pycbc_mismatch(y_pred_tr, data_tr, delta_t, metric)
        mm_ts, bst_ts, wrst_ts = pycbc_mismatch(y_pred_ts, data_ts, delta_t, metric)
        mm_val, bst_val, wrst_val = pycbc_mismatch(y_pred_val, data_val, delta_t, metric)

    perc_10_tr = get_percentile_index(mm_tr, 0.9)
    perc_50_tr = get_percentile_index(mm_tr, 0.5)

    perc_10_ts = get_percentile_index(mm_ts, 0.9)
    perc_50_ts = get_percentile_index(mm_ts, 0.5)

    perc_10_val = get_percentile_index(mm_val, 0.9)
    perc_50_val = get_percentile_index(mm_val, 0.5)

    dump(mm_tr, os.path.join(dir, "mismatches_train.bin"))
    dump(mm_ts, os.path.join(dir, "mismatches_test.bin"))
    dump(mm_val, os.path.join(dir, "mismatches_validation.bin"))

    print("Plotting histogram", end='\r')
    plot_mismatch_histogram(mm_tr, dir_tr, "Train")
    plot_mismatch_histogram(mm_ts, dir_ts, "Test")
    plot_mismatch_histogram(mm_val, dir_val, "Validation")

    print("Plotting waveforms", end='\r')
    plot_waveform_comparison(y_pred_tr[bst_tr], data_tr[bst_tr], delta_t, dir_tr, title = f"Best train case. Mismatch: {mm_tr[bst_tr]:.3e}", save_name = "best_waveform.jpg")
    plot_waveform_comparison(y_pred_ts[bst_ts], data_ts[bst_ts], delta_t, dir_ts, title = f"Best test case. Mismatch: {mm_ts[bst_ts]:.3e}", save_name = "best_waveform.jpg")
    plot_waveform_comparison(y_pred_val[bst_val], data_val[bst_val], delta_t, dir_val, title = f"Best validation case. Mismatch: {mm_val[bst_val]:.3e}", save_name = "best_waveform.jpg")
    
    plot_waveform_comparison(y_pred_tr[wrst_tr], data_tr[wrst_tr], delta_t, dir_tr, title = f"Worst train case. Mismatch: {mm_tr[wrst_tr]:.3e}", save_name = "worst_waveform.jpg")
    plot_waveform_comparison(y_pred_ts[wrst_ts], data_ts[wrst_ts], delta_t, dir_ts, title = f"Worst test case. Mismatch: {mm_ts[wrst_ts]:.3e}", save_name = "worst_waveform.jpg")
    plot_waveform_comparison(y_pred_val[wrst_val], data_val[wrst_val], delta_t, dir_val, title = f"Worst validation case. Mismatch: {mm_val[wrst_val]:.3e}", save_name = "worst_waveform.jpg")

    plot_waveform_comparison(y_pred_tr[perc_10_tr], data_tr[perc_10_tr], delta_t, dir_tr, title = f"10th percentile train waveform. Mismatch: {mm_tr[perc_10_tr]:.3e}", save_name = "perc_10_waveform.jpg")
    plot_waveform_comparison(y_pred_ts[perc_10_ts], data_ts[perc_10_ts], delta_t, dir_ts, title = f"10th percentile test waveform. Mismatch: {mm_ts[perc_10_ts]:.3e}", save_name = "perc_10_waveform.jpg")
    plot_waveform_comparison(y_pred_val[perc_10_val], data_val[perc_10_val], delta_t, dir_val, title = f"10th percentile validation waveform. Mismatch: {mm_val[perc_10_val]:.3e}", save_name = "perc_10_waveform.jpg")

    plot_waveform_comparison(y_pred_tr[perc_50_tr], data_tr[perc_50_tr], delta_t, dir_tr, title = f"50th percentile train waveform. Mismatch: {mm_tr[perc_50_tr]:.3e}", save_name = "perc_50_waveform.jpg")
    plot_waveform_comparison(y_pred_ts[perc_50_ts], data_ts[perc_50_ts], delta_t, dir_ts, title = f"50th percentile test waveform. Mismatch: {mm_ts[perc_50_ts]:.3e}", save_name = "perc_50_waveform.jpg")
    plot_waveform_comparison(y_pred_val[perc_50_val], data_val[perc_50_val], delta_t, dir_val, title = f"50th percentile validation waveform. Mismatch: {mm_val[perc_50_val]:.3e}", save_name = "perc_50_waveform.jpg")

def nb_plot_waveform_comparison(y_pred, y_true, delta_t, title):

    total_length = len(y_pred)
    times = np.arange(total_length) * delta_t

    fig, ax = plt.subplots()

    max_amp = max([np.max(np.abs(y_true)), np.max(np.abs(y_pred))])

    ax.plot(times, y_pred, 'r', linewidth = 1.5, label = "Generated waveform")
    ax.plot(times, y_true, 'k--', linewidth = 1, label = "Targeted waveform")
    plt.legend()
    plt.title(title)
    
    plt.ylabel("Strain")
    plt.xlabel("Time (s)")

    # inset axes....
    if total_length == 2048:
        x1, x2, y1, y2 = times[1850], times[-1], -max_amp*1.1, max_amp*1.1
    else:
        x1, x2, y1, y2 = times[3200], times[3800], -max_amp*1.1, max_amp*1.1

    axins = ax.inset_axes(
        [0.06, 0.73, 0.73, 0.25],
        xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    axins.plot(times, y_pred, 'r', linewidth = 1.5)
    axins.plot(times, y_true, 'k--', linewidth = 1)

    ax.indicate_inset_zoom(axins, edgecolor="black")

def nb_plot_mismatch(mismatches):

    bins = np.logspace(np.log10(min(mismatches)), np.log10(max(mismatches)), num=30)
    _ = plt.hist(mismatches, bins = bins)
    plt.grid(True, which="both", axis="both", linestyle='-', color='gray', linewidth=0.5, alpha = 0.7)
    plt.xscale('log')
    plt.title(f'Mean mismatch: {np.mean(mismatches):3e}')
    plt.ylabel('Count')
    plt.xlabel(f'Mismatch')

def nb_evaluate_model(y_pred, y_true, delta_t):

    mismatches, best_mm, worst_mm = pycbc_mismatch(y_pred, y_true, delta_t)

    nb_plot_mismatch(mismatches)
    nb_plot_waveform_comparison(y_pred[best_mm], y_true[best_mm], delta_t, f"Best case scenario. Mismatch: {mismatches[best_mm]:.3e}")
    nb_plot_waveform_comparison(y_pred[worst_mm], y_true[worst_mm], delta_t, f"Worst case scenario. Mismatch: {mismatches[worst_mm]:.3e}")