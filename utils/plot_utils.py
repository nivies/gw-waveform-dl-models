import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
from joblib import dump
from pycbc.filter.matchedfilter import match
from pycbc.types import TimeSeries
from tqdm import tqdm

def get_percentile_index(mm, percentile):
    sorted_array = np.sort(mm)
    perc_sorted_idx = int(percentile * len(np.sort(mm)))
    value_perc_10_tr = sorted_array[perc_sorted_idx]
    perc_idx = np.where(mm == value_perc_10_tr)[0][0]

    return perc_idx

def make_plotting_dirs(dir):
    
    if not os.path.isdir(dir):
        os.mkdir(dir)

    dir_train = os.path.join(dir, "train/")
    dir_test = os.path.join(dir, "test/")

    if not os.path.isdir(dir_train):
        os.mkdir(dir_train)
    if not os.path.isdir(dir_test):
        os.mkdir(dir_test)
    
    return dir_train, dir_test
    
def pycbc_mismatch(y_pred, y_true, delta_t, real = True):

    '''
    delta_t must be in seconds
    '''

    mm = []
    for prd, grnd in tqdm(zip(y_pred, y_true), total = len(y_pred), desc = "Calculating mismatches"):
        if real:
            pred_ts = TimeSeries(np.real(prd).astype(float), delta_t)
            grnd_ts = TimeSeries(np.real(grnd).astype(float), delta_t)
        else:
            pred_ts = TimeSeries(np.imag(prd).astype(float), delta_t)
            grnd_ts = TimeSeries(np.imag(grnd).astype(float), delta_t)
        mm.append(1-match(pred_ts, grnd_ts)[0])
    
    return np.array(mm), np.argmin(mm), np.argmax(mm)

def plot_mismatch_histogram(mismatches, dir, train):

    if train:
        train_test = "Train"
    else:
        train_test = "Test"

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

def make_plots(model, dir, data_loader):

    dir_tr, dir_ts = make_plotting_dirs(dir)

    y_pred_tr = model.predict(data_loader.X_train, batch_size = 1024)
    y_pred_ts = model.predict(data_loader.X_test, batch_size = 1024)

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