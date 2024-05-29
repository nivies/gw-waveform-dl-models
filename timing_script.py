import timeit
import argparse
import tensorflow as tf
import os
import h5py
from keras.models import load_model
import numpy as np
from tqdm import tqdm

def main():

    parser = argparse.ArgumentParser(description='Script for timing generation times from different algorithms.')

    parser.add_argument('-gm', '--generation_method', dest='method', help='Method for GW generation: dense, mapped, regularized, cvae or surrogate.', metavar='')
    parser.add_argument('-pu', '--processing_unit', dest='pu', help='Whether to run the generation method on cpu (\'cpu\') or on gpu (\'gpu\').', metavar='')
    parser.add_argument('-bs', '--batch_size', dest='batch_size', help='Batch size for predict method for NN based generation methods.', metavar='')
    parser.add_argument('-d', '--dataset', dest='data', help='Dataset to run the script on.\n1 -> Mass BNS\n2 -> Mass-lambda BNS\n3 -> Mass BBH\n4 -> Mass z-spin BBH\n5 -> Mass full-spin BBH\n6 -> Osvaldo\'s dataset (z-spin BBH)', metavar='')
    parser.add_argument('-n', '--n_datapoints', dest='N', help='Number of waveforms to generate.', metavar='')
    parser.add_argument('-en', '--execution_number', dest='execution_number', help='Number of script executions for timing.', default = 5, metavar='')

    args = parser.parse_args()

    data_selector = {
        '1': "/home/nino/GW/data/bns_mass.hdf",
        '2': "/home/nino/GW/data/bns_mass_lbd.hdf",
        '3': "/home/nino/GW/data/bbh_q_data.hdf5",
        '4': "/home/nino/GW/data/bbh_qz_data.hdf5",
        '5': "/home/nino/GW/data/bbh_qzp_data.hdf5",
        '6': "/home/nino/GW/Keras-Project-Template/data/nonhyb_noprec_dataset_q8.hdf"
    }

    model_selector = {
        "dense1": "/home/nino/GW/Keras-Project-Template/experiments/dense_q_bns/gw_dense_q_bns/checkpoints",
        "dense2": "/home/nino/GW/Keras-Project-Template/experiments/dense_q_lbd_bns/gw_dense_q_lbd_bns/checkpoints",
        "dense3": "/home/nino/GW/Keras-Project-Template/experiments/q_bbh_dense_mal/gw_dense_q_bbh/checkpoints",
        "dense4": "/home/nino/GW/Keras-Project-Template/experiments/qz_bbh_dense_mal/gw_dense_qz_bbh/checkpoints",
        "dense5": "/home/nino/GW/Keras-Project-Template/experiments/qzp_bbh_dense_mal/gw_dense_qzp_bbh/checkpoints",
        "mapped1": "/home/nino/GW/Keras-Project-Template/experiments/mapper_q_bns_def/mapper_q_bns/checkpoints",
        "mapped2": "/home/nino/GW/Keras-Project-Template/experiments/mapper_q_lbd_bns/mapper_q_lbd_bns/checkpoints",
        "mapped3": "/home/nino/GW/Keras-Project-Template/experiments/mapper_q_bbh/gw_mapped_generator_q/checkpoints",
        "mapped4": "/home/nino/GW/Keras-Project-Template/experiments/mapper_qz_bbh/mapper_qz_bbh/checkpoints",
        "mapped5": "/home/nino/GW/Keras-Project-Template/experiments/mapper_qzp_bbh/mapper_qzp_bbh/checkpoints",
        "regularized1": "/home/nino/GW/Keras-Project-Template/experiments/reg_q_bns/reg_q_bns/checkpoints",
        "regularized2": "/home/nino/GW/Keras-Project-Template/experiments/reg_q_lbd_bns/reg_q_lbd_bns/checkpoints",
        "regularized3": "/home/nino/GW/Keras-Project-Template/experiments/q_bbh_reg_light/reg_q_bbh_light/checkpoints",
        "regularized4": "/home/nino/GW/Keras-Project-Template/experiments/reg_qz_bbh_def/reg_qz_bbh/checkpoints",
        "regularized5": "/home/nino/GW/Keras-Project-Template/experiments/reg_qzp_bbh_good/reg_qzp_bbh/checkpoints",
        "light_amp6" : "/home/nino/GW/Keras-Project-Template/experiments/osv_amp_phs_overlap/reg_q_bbh_light/checkpoints"
    }

    def code_generator(method, data, N, bs = None, pu = None):

        if pu == 'cpu':
            os.environ["CUDA_VISIBLE_DEVICES"]=""
        elif pu == 'gpu':
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
        if method == "surrogate":
            dataset = h5py.File(data_selector[data])

            if data == '1':
                from pycbc.waveform import get_td_waveform

                m1, m2 = dataset['parameters'][:N, 0], dataset['parameters'][:N, 1]
                
                M_sun = 1.9885e30
                c = 299792458
                G = 6.67430e-11
                delta_t = 2*G*M_sun/(c**3)

                code_to_test = f'''for m1_, m2_ in zip(m1, m2):
    get_td_waveform(approximant="IMRPhenomPv2_NRTidal", mass1=m1_, mass2=m2_, f_lower=100, delta_t = {delta_t}, inclination=0.0, distance=100)
                ''' 

                variables_dict = {'m1': m1, 'm2': m2, 'get_td_waveform': get_td_waveform}


                return code_to_test, variables_dict

            elif data == '2':
                from pycbc.waveform import get_td_waveform

                m1, m2, lb1, lb2 = dataset['parameters'][:N, 0], dataset['parameters'][:N, 1], dataset['parameters'][:N, 2], dataset['parameters'][:N, 3]


                M_sun = 1.9885e30
                c = 299792458
                G = 6.67430e-11
                delta_t = 2*G*M_sun/(c**3)

                code_to_test = f'''for m1_, m2_, lb1_, lb2_ in zip(m1, m2, lb1, lb2):
    get_td_waveform(approximant="IMRPhenomPv2_NRTidal", mass1=m1_, mass2=m2_, lambda1 = lb1_, lambda2 = lb2_, f_lower=100, delta_t = {delta_t}, inclination=0.0, distance=100)
                '''

                variables_dict = {'m1': m1, 'm2': m2, 'lb1': lb1, 'lb2': lb2, 'get_td_waveform': get_td_waveform}

                return code_to_test, variables_dict

            elif data == '3':
                import gwsurrogate
                sur = gwsurrogate.LoadSurrogate('NRSur7dq4')
                m1, m2 = dataset['parameters'][:N, 0], dataset['parameters'][:N, 1]
                q = m1/m2

                code_to_test = """for q_ in q:
    sur(q_, [0, 0, 0], [0, 0, 0], dt = 2, f_low = 0)
                """

                variables_dict = {'q': q, 'sur': sur}

                return code_to_test, variables_dict

            elif data == '4':
                import gwsurrogate
                sur = gwsurrogate.LoadSurrogate('NRSur7dq4')

                q, z1, z2 = dataset['parameters'][:N, 0], dataset['parameters'][:N, 1], dataset['parameters'][:N, 2]

                code_to_test = """for q_, z1_, z2_ in zip(q, z1, z2):
    sur(q_, [0, 0, z1_], [0, 0, z2_], dt = 2, f_low = 0)
                """

                variables_dict = {'q': q, 'z1': z1, 'z2': z2, 'sur': sur}

                return code_to_test, variables_dict
            
            elif data == '5':
                import gwsurrogate
                sur = gwsurrogate.LoadSurrogate('NRSur7dq4')

                # global q, s1x, s1y, s1z, s2x, s2y, s2z

                q, s1x, s1y, s1z, s2x, s2y, s2z = dataset['parameters'][:N, 0], dataset['parameters'][:N, 1], dataset['parameters'][:N, 2], dataset['parameters'][:N, 3], dataset['parameters'][:N, 4], dataset['parameters'][:N, 5], dataset['parameters'][:, 6]
                code_to_test = """for q_, s1x_, s1y_, s1z_, s2x_, s2y_, s2z_ in zip(q, s1x, s1y, s1z, s2x, s2y, s2z):
    sur(q_, [s1x_, s1y_, s1z_], [s2x_, s2y_, s2z_], dt = 2, f_low = 0)
                """

                variables_dict = {'q': q, 's1x': s1x, 's1y': s1y, 's1z': s1z, 's2x': s2x, 's2y': s2y, 's2z': s2z, 'sur': sur}

                return code_to_test, variables_dict


        elif method == "cvae":
            dataset = h5py.File(data_selector['5'])

            z_samples = np.random.normal(0, 1, (N, 10))
            dec = load_model("/home/nino/GW/Keras-Project-Template/experiments/cvae/decoder_cvae")
            params = dataset['parameters'][:N]
            code_to_test=f'''dec.predict([z_samples, params], batch_size={bs}, verbose=0)
            '''

            variables_dict = {'code_to_test': code_to_test, 'dec': dec, 'params': params, 'z_samples': z_samples}

            return code_to_test, variables_dict

        else:
            from utils.eval import load_model_and_data_loader

            model, data_loader = load_model_and_data_loader(model_selector[method + data])

            code_to_test = f'''model.predict(data_loader.X_train[:{N}], batch_size={bs}, verbose=0)
            '''

            variables_dict = {'data_loader': data_loader, 'model': model}

            return code_to_test, variables_dict

    code, variables_dict = code_generator(method = args.method, data = args.data, pu = args.pu, N = int(args.N), bs = args.batch_size)

    execution_time = [timeit.timeit(code, globals = variables_dict, number = 1) for _ in tqdm(range(int(args.execution_number)+1), desc = "Running generative function")]
    execution_time = execution_time[1:]
    print("\n\n")
    print(execution_time)
    print("\n\n")
    print(f"Execution mean time for generating {args.N} waveforms: {np.mean(execution_time):.5e} seconds.\n Standard deviation: {np.std(execution_time):.5e} seconds. Run {args.execution_number} times.")

if __name__ == '__main__':
    main()