import argparse
import h5py
import numpy as np
import tensorflow as tf
import nvidia_smi

def query_gpu():
    nvidia_smi.nvmlInit()

    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    print("\n")
    print(f"Total memory: {info.total*10**(-9):.3f} Gb.")
    print(f"Free memory: {info.free*10**(-9):.3f} Gb.")
    print(f"Used memory: {info.used*10**(-9):.3f} Gb.")
    print("\n")

    nvidia_smi.nvmlShutdown()

def configure_device(use_cpu=False, memory_growth=True, gpu_memory_limit=None):
    
    if use_cpu:
        print("Using CPU...")
        tf.config.set_visible_devices([], 'GPU')  # Disable GPU, force TensorFlow to use CPU
    else:
        # List available GPUs
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"Found {len(gpus)} GPU(s), configuring memory settings...")

            for gpu in gpus:
                if memory_growth:
                    try:
                        # Set memory growth if enabled
                        tf.config.experimental.set_memory_growth(gpu, True)
                        print(f"Enabled memory growth for GPU: {gpu}")
                    except RuntimeError as e:
                        print(f"Error setting memory growth: {e}")
                
                if gpu_memory_limit:
                    try:
                        # Set a specific GPU memory limit if provided
                        tf.config.set_logical_device_configuration(
                            gpu,
                            [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_memory_limit)])
                        print(f"Set memory limit of {gpu_memory_limit} MB for GPU: {gpu}")
                    except RuntimeError as e:
                        print(f"Error setting memory limit: {e}")
        else:
            print("No GPU found, falling back to CPU...")
            tf.config.set_visible_devices([], 'GPU')  # No GPU available, fallback to CPU


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args

def split(x, y, split):

    split_idx = np.floor(len(x)*split).astype(int)

    x_tr = x[:split_idx]
    x_ts = x[split_idx:]
    y_tr = y[:split_idx]
    y_ts = y[split_idx:]

    return x_tr, x_ts, y_tr, y_ts

def load_sxs_data(split_n):
    
    f = h5py.File("/home/nino/GW/data/sxs_clean.h5")
    x = f['parameters'][:]
    y = f['waveforms'][:]

    q_mask = [i<2.5 for i in x[:, 0]]
    low_p_mask = [True if (x1 < 0.1 and y1 < 0.1 and x2 < 0.1 and y2 < 0.1) else False for x1, x2, y1, y2 in zip(np.abs(x[:, 1]), np.abs(x[:, 2]), np.abs(x[:, 4]), np.abs(x[:, 5]))]
    med_p_mask = [True if (0.1 < x1 < 0.3 or 0.1 < y1 < 0.3 or  0.1 < x2 < 0.3 or 0.1 < y2 < 0.3) else False for x1, x2, y1, y2 in zip(np.abs(x[:, 1]), np.abs(x[:, 2]), np.abs(x[:, 4]), np.abs(x[:, 5]))]
    high_p_mask = [True if (x1 > 0.3 or y1 > 0.3 or x2 > 0.3 or y2 > 0.3) else False for x1, x2, y1, y2 in zip(np.abs(x[:, 1]), np.abs(x[:, 2]), np.abs(x[:, 4]), np.abs(x[:, 5]))]

    low_z_mask = [True if (z1 < 0.1 and z2 < 0.1) else False for z1, z2 in zip(x[:, 3], x[:, 6])]
    high_z_mask = [True if (z1 > 0.1 or z2 > 0.1) else False for z1, z2 in zip(x[:, 3], x[:, 6])]

    lll_mask = [q and lz and pz for q, lz, pz in zip(q_mask, low_z_mask, low_p_mask)]
    llm_mask = [q and lz and pz for q, lz, pz in zip(q_mask, low_z_mask, med_p_mask)]
    llh_mask = [q and lz and pz for q, lz, pz in zip(q_mask, low_z_mask, high_p_mask)]

    lhl_mask = [q and lz and pz for q, lz, pz in zip(q_mask, high_z_mask, low_p_mask)]
    lhm_mask = [q and lz and pz for q, lz, pz in zip(q_mask, high_z_mask, med_p_mask)]
    lhh_mask = [q and lz and pz for q, lz, pz in zip(q_mask, high_z_mask, high_p_mask)]

    hll_mask = [not q and lz and pz for q, lz, pz in zip(q_mask, low_z_mask, low_p_mask)]
    hlm_mask = [not q and lz and pz for q, lz, pz in zip(q_mask, low_z_mask, med_p_mask)]
    hlh_mask = [not q and lz and pz for q, lz, pz in zip(q_mask, low_z_mask, high_p_mask)]

    hhl_mask = [not q and lz and pz for q, lz, pz in zip(q_mask, high_z_mask, low_p_mask)]
    hhm_mask = [not q and lz and pz for q, lz, pz in zip(q_mask, high_z_mask, med_p_mask)]
    hhh_mask = [not q and lz and pz for q, lz, pz in zip(q_mask, high_z_mask, high_p_mask)]

    split_ratio = split_n/1037

    x_lll_train, x_lll_test, y_lll_train, y_lll_test = split(x[lll_mask], y[lll_mask], split=split_ratio)
    x_llm_train, x_llm_test, y_llm_train, y_llm_test = split(x[llm_mask], y[llm_mask], split=split_ratio)
    x_llh_train, x_llh_test, y_llh_train, y_llh_test = split(x[llh_mask], y[llh_mask], split=split_ratio)

    x_lhl_train, x_lhl_test, y_lhl_train, y_lhl_test = split(x[lhl_mask], y[lhl_mask], split=split_ratio)
    x_lhm_train, x_lhm_test, y_lhm_train, y_lhm_test = split(x[lhm_mask], y[lhm_mask], split=split_ratio)
    x_lhh_train, x_lhh_test, y_lhh_train, y_lhh_test = split(x[lhh_mask], y[lhh_mask], split=split_ratio)

    x_hll_train, x_hll_test, y_hll_train, y_hll_test = split(x[hll_mask], y[hll_mask], split=split_ratio)
    x_hlm_train, x_hlm_test, y_hlm_train, y_hlm_test = split(x[hlm_mask], y[hlm_mask], split=split_ratio)
    x_hlh_train, x_hlh_test, y_hlh_train, y_hlh_test = split(x[hlh_mask], y[hlh_mask], split=split_ratio)

    x_hhl_train, x_hhl_test, y_hhl_train, y_hhl_test = split(x[hhl_mask], y[hhl_mask], split=split_ratio)
    x_hhm_train, x_hhm_test, y_hhm_train, y_hhm_test = split(x[hhm_mask], y[hhm_mask], split=split_ratio)
    x_hhh_train, x_hhh_test, y_hhh_train, y_hhh_test = split(x[hhh_mask], y[hhh_mask], split=split_ratio)

    x_train = np.concatenate([x_lll_train, x_llm_train, x_llh_train, x_lhl_train, x_lhm_train, x_lhh_train, x_hll_train, x_hlm_train, x_hlh_train, x_hhl_train, x_hhm_train, x_hhh_train], axis = 0)
    y_train = np.concatenate([y_lll_train, y_llm_train, y_llh_train, y_lhl_train, y_lhm_train, y_lhh_train, y_hll_train, y_hlm_train, y_hlh_train, y_hhl_train, y_hhm_train, y_hhh_train], axis = 0)
    x_test = np.concatenate([x_lll_test, x_llm_test, x_llh_test, x_lhl_test, x_lhm_test, x_lhh_test, x_hll_test, x_hlm_test, x_hlh_test, x_hhl_test, x_hhm_test, x_hhh_test], axis = 0)
    y_test = np.concatenate([y_lll_test, y_llm_test, y_llh_test, y_lhl_test, y_lhm_test, y_lhh_test, y_hll_test, y_hlm_test, y_hlh_test, y_hhl_test, y_hhm_test, y_hhh_test], axis = 0)

    return x_train, y_train, x_test, y_test