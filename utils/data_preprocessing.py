import numpy as np
import h5py

def unwrap_phase(complex_array):
    phase = np.angle(complex_array)
    unwrapped_phase = np.unwrap(phase)
    return unwrapped_phase

def get_phases(array):
    out = unwrap_phase(array)
    out = out-out[..., np.newaxis,0]
    return out.astype(float)

def load_data_(data, output_type):
    
    flag_delta_t = False

    amplitudes = np.abs(data['waveforms'][:])
    phases = get_phases(data['waveforms'][:])

    try:
        delta_t = data['waveforms'].attrs['delta_t_seconds']
        flag_delta_t = True
    except:
        Warning("No delta_t provided in dataset.")

    if flag_delta_t:
        if output_type == 'hp':
            return data['parameters'][:], amplitudes*np.cos(phases), delta_t
        elif output_type == 'hc':
            return data['parameters'][:], amplitudes*np.sin(phases), delta_t
        elif output_type == 'amplitude_phase':
            return data['parameters'][:], np.concatenate([amplitudes, phases], axis = 1), delta_t
        elif output_type == 'complex':
            return data['parameters'][:], amplitudes*np.cos(phases) + 1.0j*amplitudes*np.sin(phases), delta_t
        else:
            NameError("output_type specified not implemented.")
    else:
        if output_type == 'hp':
            return data['parameters'][:], amplitudes*np.cos(phases)
        elif output_type == 'hc':
            return data['parameters'][:], amplitudes*np.sin(phases)
        elif output_type == 'amplitude_phase':
            return data['parameters'][:], np.concatenate([amplitudes, phases])
        elif output_type == 'complex':
            return data['parameters'][:], amplitudes*np.cos(phases) + 1.0j*amplitudes*np.sin(phases)
        else:
            NameError("output_type specified not implemented.")

def load_sxs_data_(data, output_type):
    
    flag_delta_t = False

    amplitudes_sxs = np.abs(data['sxs']['waveforms'][:])
    phases_sxs = get_phases(data['sxs']['waveforms'][:])

    amplitudes_sur = np.abs(data['surrogate']['waveforms'][:])
    phases_sur = get_phases(data['surrogate']['waveforms'][:])

    try:
        delta_t_sxs = data['sxs']['waveforms'].attrs['delta_t_seconds']
        delta_t_sur = data['surrogate']['waveforms'].attrs['delta_t_seconds']
        if delta_t_sxs == delta_t_sur:
            delta_t = delta_t_sxs
            flag_delta_t = True
        else:
            ValueError("Delta t for numerical relativity and surrogate waveforms must be equal.")
    except:
        Warning("No delta_t provided in dataset.")

    if flag_delta_t:
        if output_type == 'hp':
            return (data['sxs']['parameters'][:], amplitudes_sxs*np.cos(phases_sxs)), (data['surrogate']['parameters'][:], amplitudes_sur*np.cos(phases_sur)), delta_t
        elif output_type == 'hc':
            return (data['sxs']['parameters'][:], amplitudes_sxs*np.sin(phases_sxs)), (data['surrogate']['parameters'][:], amplitudes_sur*np.sin(phases_sur)), delta_t
        elif output_type == 'amplitude_phase':
            return (data['sxs']['parameters'][:], np.concatenate([amplitudes_sxs, phases_sxs])), (data['surrogate']['parameters'][:], np.concatenate([amplitudes_sur, phases_sur])), delta_t
        else:
            NameError("output_type specified not implemented.")
    else:
        if output_type == 'hp':
            return (data['sxs']['parameters'][:], amplitudes_sxs*np.cos(phases_sxs)), (data['surrogate']['parameters'][:], amplitudes_sur*np.cos(phases_sur))
        elif output_type == 'hc':
            return (data['sxs']['parameters'][:], amplitudes_sxs*np.sin(phases_sxs)), (data['surrogate']['parameters'][:], amplitudes_sur*np.sin(phases_sur))
        elif output_type == 'amplitude_phase':
            return (data['sxs']['parameters'][:], np.concatenate([amplitudes_sxs, phases_sxs])), (data['surrogate']['parameters'][:], np.concatenate([amplitudes_sur, phases_sur]))
        else:
            NameError("output_type specified not implemented.")


def load_data(path, output_type):
    data = h5py.File(path)

    if 'waveforms' in data.keys():
        return load_data_(data, output_type)
    else:
        return load_sxs_data_(data, output_type)