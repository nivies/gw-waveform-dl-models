import numpy as np
from utils.utils import load_sxs_data
from joblib import dump
import matplotlib.pyplot as plt
from keras.models import load_model
import h5py
from pycbc.filter.matchedfilter import match
from pycbc.types import TimeSeries
from tqdm import tqdm
import tensorflow as tf
import os

cpu = False

if cpu:
  os.environ["CUDA_VISIBLE_DEVICES"]=""
else:
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])
    except RuntimeError as e:
      print(e)

def pycbc_mismatch(y_pred, y_true, delta_t, real = True):

    '''
    delta_t must be in seconds
    '''

    mm = []
    for prd, grnd in zip(y_pred, y_true):
        if real:
            pred_ts = TimeSeries(np.real(prd).astype(float), delta_t)
            grnd_ts = TimeSeries(np.real(grnd).astype(float), delta_t)
        else:
            pred_ts = TimeSeries(np.imag(prd).astype(float), delta_t)
            grnd_ts = TimeSeries(np.imag(grnd).astype(float), delta_t)
        mm.append(1-match(pred_ts, grnd_ts)[0])
    
    return np.array(mm), np.argmin(mm), np.argmax(mm)

def get_data(par, y, N):
    z_samples = np.random.normal(0, 1, (N, 10))
    pars = np.tile(par, (N, 1))
    y_ground = np.tile(y, (N, 1))

    return z_samples, pars, y_ground

def plot_mean_distribution(dec, y_ground, pars, N, delta_t, dir, train):

    if not os.path.isdir(dir):
       os.mkdir(dir)

    means = []
    for par, y in tqdm(zip(pars, y_ground), total = len(y_ground), desc = "Getting distribution of means"):
        z_samples, params, y_target = get_data(par, y, N)
        y_pred = dec.predict([z_samples, params], batch_size = np.floor(N/10).astype(int), verbose = 0)
        mismatch, _, _ = pycbc_mismatch(y_pred, y_target, delta_t, real = True)
        means.append(np.mean(mismatch))
    
    
    bins = np.logspace(np.log10(min(means)), np.log10(max(means)), num=30)
    plt.hist(means, bins = bins)
    plt.title(f"Distribution of means for cVAE generator\nMean of means: {np.mean(means):.3e}")
    plt.grid(True, which="both", axis="both", linestyle='-', color='gray', linewidth=0.5, alpha = 0.7)
    plt.xscale('log')
    plt.ylabel("Count")
    plt.xlabel("Mismatch mean")
    if train:
      opt_dir_plot = os.path.join(dir, "train_mismatch_comparison.jpg")
      opt_dir_mm = os.path.join(dir, "mean_train_mismatches.bin")

      dump(np.array(means), opt_dir_mm)
      plt.savefig(opt_dir_plot)
    else:
      opt_dir_plot = os.path.join(dir, "test_mismatch_comparison.jpg")
      opt_dir_mm = os.path.join(dir, "mean_test_mismatches.bin")

      dump(np.array(means), opt_dir_mm)
      plt.savefig(opt_dir_plot)
      plt.clf()
    

dec = load_model("/home/nino/GW/Keras-Project-Template/experiments/cvae/decoder_cvae")
f_sur = h5py.File("/home/nino/GW/data/bbh_qzp_data.hdf5")
# sxs_x_train, sxs_y_train, sxs_x_test, sxs_y_test = load_sxs_data(700)
# sxs_y_train = np.real(sxs_y_train)
# sxs_y_test = np.real(sxs_y_test)

plot_mean_distribution(dec, f_sur['waveforms'][80000:], f_sur['parameters'][80000:], 100, f_sur['waveforms'].attrs['delta_t_seconds'], "/home/nino/GW/Keras-Project-Template/figures/cvae_qzp", train = False)
plot_mean_distribution(dec, f_sur['waveforms'][:80000], f_sur['parameters'][:80000], 100, f_sur['waveforms'].attrs['delta_t_seconds'], "/home/nino/GW/Keras-Project-Template/figures/cvae_qzp", train = True)