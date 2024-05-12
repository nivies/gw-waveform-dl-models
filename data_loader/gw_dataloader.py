import numpy as np
from sklearn.utils import shuffle
from keras.utils import Sequence
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from base.base_data_loader import BaseDataLoader
from utils.data_preprocessing import *

'''
File for declaring data generators, data loaders and some utilities such as train-test splitting, input parameter scaling and sample weighting
'''

shuffle_seed = 42

def split_data(x, y, split_size):
    
    split_idx = np.floor(x.shape[0]*(1 - split_size)).astype(int)
    return (x[:split_idx], y[:split_idx]), (x[split_idx:], y[split_idx:])

class DataGenerator(Sequence):

    '''
    General data generator class for faster data loading.

    Input parameters:
    -----------------

    x_set: numpy array
        Input data.
    
    y_set: numpy array
        Output data.

    batch_size: int
        Training/validation batch size.

    sample_weights: int
        Value for sample weighting.
    -----------------

    Output:
        A keras' Sequence object for quick data loading into the keras' models fit method.
    
    '''

    def __init__(self, x_set, y_set, batch_size, sample_weights = None):
        
        if sample_weights is not None:
            self.x, self.y, self.sample_weights = shuffle(x_set, y_set, sample_weights, random_state=37)
        else:
            self.x, self.y = shuffle(x_set, y_set, random_state=37)
            self.sample_weights = sample_weights
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.sample_weights is not None:
            return batch_x, batch_y, self.sample_weights[idx * self.batch_size:(idx + 1) * self.batch_size]
        else:
            return batch_x, batch_y
    
class MultiOutputDataGenerator(Sequence):

    '''
    General data generator class for faster data loading. It allows for trainings with multiple outputs (regularized autoencoder).

    Input parameters:
    -----------------

    x: numpy array
        Input data.
    
    y: numpy array
        Output data.

    batch_size: int
        Training/validation batch size.

    sample_weights: int
        Value for sample weighting.
    -----------------

    Output:
        A keras' Sequence object for quick data loading into the keras' models fit method.
    
    '''

    def __init__(self, x, y, batch_size, sample_weights = None):

        if sample_weights is not None:
            self.x, self.y_0, self.y_1, self.sample_weights = shuffle(x, y[0], y[1], sample_weights, random_state=37)
        else:
            self.x, self.y_0, self.y_1 = shuffle(x, y[0], y[1], random_state=37)
            self.sample_weights = sample_weights
        self.batch_size = batch_size

        self.y = [self.y_0, self.y_1]

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = [output[idx * self.batch_size:(idx + 1) * self.batch_size] for output in self.y]

        if self.sample_weights is not None:
            return np.array(batch_x), [np.array(output) for output in batch_y], self.sample_weights[idx * self.batch_size:(idx + 1) * self.batch_size]
        else:
            return np.array(batch_x), [np.array(output) for output in batch_y]
    
class MultiInputDataGenerator(Sequence):

    '''
    General data generator class for faster data loading. It allows for trainings with multiple inputs (cVAE).

    Input parameters:
    -----------------

    x: numpy array
        Input data.
    
    y: numpy array
        Output data.

    batch_size: int
        Training/validation batch size.

    sample_weights: int
        Value for sample weighting.
    -----------------

    Output:
        A keras' Sequence object for quick data loading into the keras' models fit method.
    
    '''

    def __init__(self, x, y, batch_size, sample_weights = None):

        if sample_weights is not None:
            self.x_0, self.x_1, self.y, self.sample_weights = shuffle(x[0], x[1], y, sample_weights, random_state=37)
        else:
            self.x_0, self.x_1, self.y = shuffle(x[0], x[1], y, random_state=37)
            self.sample_weights = sample_weights
        self.batch_size = batch_size

        self.x= [self.x_0, self.x_1]

    def __len__(self):
        return int(np.ceil(len(self.y) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = [input_[idx * self.batch_size:(idx + 1) * self.batch_size] for input_ in self.x]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.sample_weights is not None:
            return [np.array(input_) for input_ in batch_x], np.array(batch_y),  self.sample_weights[idx * self.batch_size:(idx + 1) * self.batch_size]
        else:
            return [np.array(input_) for input_ in batch_x], np.array(batch_y)

class GWDataLoader(BaseDataLoader):

    '''
    Data loader class for dense models. Loads data, splits in train and test sets, allows for input scaling and SXS augmented dataset 
    loading. After data loading, calls the data generator classes.

    Input parameters:
    -----------------

    config: dict
        Configuration dictionary created from the parsed data of the .json file.
    '''

    def __init__(self, config):
        super(GWDataLoader, self).__init__(config)

        self.train_weights = None

        load_data_opt = self.get_data()
        
        if len(load_data_opt) == 2:
            (self.X_train, self.y_train), (self.X_test, self.y_test) = load_data_opt
        else:
            (self.X_train, self.y_train), (self.X_test, self.y_test), (self.X_sxs_train, self.y_sxs_train), (self.X_sxs_test, self.y_sxs_test), (self.X_sur_train, self.y_sur_train), (self.X_sur_test, self.y_sur_test) = load_data_opt

        if self.config.data_loader.scale_data:
            scaler = StandardScaler()
            self.X_train_, self.X_test_ = deepcopy(self.X_train), deepcopy(self.X_test)
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)
            
        self.X, self.y = np.concatenate([self.X_train, self.X_test]), np.concatenate([self.y_train, self.y_test])
        self.in_out_shapes = {'input_shape' : self.X.shape[1], 'output_shape' : self.y.shape[1]}

    def get_train_data(self):
        if self.train_weights is not None:
            return DataGenerator(self.X_train, self.y_train, self.config.trainer.batch_size, self.train_weights)
        else:
            return DataGenerator(self.X_train, self.y_train, self.config.trainer.batch_size)
        
    def get_test_data(self):
        return DataGenerator(self.X_test, self.y_test, self.config.trainer.batch_size_test)

    def get_data(self):

        load_data_opt = load_data(self.config.data_loader.data_path, self.config.data_loader.data_output_type)
        split_size = self.config.data_loader.split_size

        if len(load_data_opt)==3:
            opt1, opt2, self.delta_t = load_data_opt
        else:
            opt1, opt2 = load_data_opt

        if len(opt1) == 2:
            x_sxs = opt1[0]
            y_sxs = opt1[1]
            x_sur = opt2[0]
            y_sur = opt2[1]

            (x_sxs_train, y_sxs_train), (x_sxs_test, y_sxs_test) = split_data(x_sxs, y_sxs, split_size)
            (x_sur_train, y_sur_train), (x_sur_test, y_sur_test) = split_data(x_sur, y_sur, split_size)

            x_train, y_train = np.concatenate([x_sur_train, x_sxs_train]), np.concatenate([y_sur_train, y_sxs_train])
            x_test, y_test = np.concatenate([x_sur_test, x_sxs_test]), np.concatenate([y_sur_test, y_sxs_test])

            self.train_weights = np.concatenate([np.ones(len(x_sur_train)), self.config.data_loader.sxs_sample_weight*np.ones(len(x_sxs_train))])

            return (x_train, y_train), (x_test, y_test), (x_sxs_train, y_sxs_train), (x_sxs_test, y_sxs_test), (x_sur_train, y_sur_train), (x_sur_test, y_sur_test)

        else:
            x = opt1
            y = opt2

            return split_data(x, y, split_size)

class GWcVAEDataLoader(BaseDataLoader):

    '''
    Data loader class for cVAE models. Loads data, splits in train and test sets, allows for input scaling and SXS augmented dataset 
    loading. After data loading, calls the multi input data generator classes.

    Input parameters:
    -----------------

    config: dict
        Configuration dictionary created from the parsed data of the .json file.
    '''

    def __init__(self, config):
        super(GWcVAEDataLoader, self).__init__(config)

        self.train_weights = None

        load_data_opt = self.get_data()
        
        if len(load_data_opt) == 2:
            (self.X_train, self.y_train), (self.X_test, self.y_test) = load_data_opt
        else:
            (self.X_train, self.y_train), (self.X_test, self.y_test), (self.X_sxs_train, self.y_sxs_train), (self.X_sxs_test, self.y_sxs_test), (self.X_sur_train, self.y_sur_train), (self.X_sur_test, self.y_sur_test) = load_data_opt

        if self.config.data_loader.scale_data:
            scaler = StandardScaler()
            self.X_train_, self.X_test_ = deepcopy(self.X_train), deepcopy(self.X_test)
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)
            
        self.X, self.y = np.concatenate([self.X_train, self.X_test]), np.concatenate([self.y_train, self.y_test])
        self.in_out_shapes = {'input_shape' : self.X.shape[1], 'output_shape' : self.y.shape[1]}

    def get_train_data(self):
        if self.train_weights is not None:
            return MultiInputDataGenerator([self.y_train, self.X_train], self.y_train, self.config.trainer.batch_size, self.train_weights)
        else:
            return MultiInputDataGenerator([self.y_train, self.X_train], self.y_train, self.config.trainer.batch_size)
        
    def get_test_data(self):
        return MultiInputDataGenerator([self.y_test, self.X_test], self.y_test, self.config.trainer.batch_size_test)

    def get_data(self):

        load_data_opt = load_data(self.config.data_loader.data_path, self.config.data_loader.data_output_type)
        split_size = self.config.data_loader.split_size

        if len(load_data_opt)==3:
            opt1, opt2, self.delta_t = load_data_opt
        else:
            opt1, opt2 = load_data_opt

        if len(opt1) == 2:
            x_sxs = opt1[0]
            y_sxs = opt1[1]
            x_sur = opt2[0]
            y_sur = opt2[1]

            (x_sxs_train, y_sxs_train), (x_sxs_test, y_sxs_test) = split_data(x_sxs, y_sxs, split_size)
            (x_sur_train, y_sur_train), (x_sur_test, y_sur_test) = split_data(x_sur, y_sur, split_size)

            x_train, y_train = np.concatenate([x_sur_train, x_sxs_train]), np.concatenate([y_sur_train, y_sxs_train])
            x_test, y_test = np.concatenate([x_sur_test, x_sxs_test]), np.concatenate([y_sur_test, y_sxs_test])

            self.train_weights = np.concatenate([np.ones(len(x_sur_train)), self.config.data_loader.sxs_sample_weight*np.ones(len(x_sxs_train))])

            return (x_train, y_train), (x_test, y_test), (x_sxs_train, y_sxs_train), (x_sxs_test, y_sxs_test), (x_sur_train, y_sur_train), (x_sur_test, y_sur_test)

        else:
            x = opt1
            y = opt2

            return split_data(x, y, split_size)

class UMAPDataLoader(BaseDataLoader):

    '''
    Data loader class for regularized autoencoder models. Loads data, splits in train and test sets, allows for input scaling and SXS augmented dataset 
    loading. Functions for including the latent space data and the UMAP projections are implemented. After data loading, calls the necessary data 
    generator classes for each of the fit calls.

    Input parameters:
    -----------------

    config: dict
        Configuration dictionary created from the parsed data of the .json file.
    '''

    def __init__(self, config):
        super(UMAPDataLoader, self).__init__(config)

        self.train_weights = None

        load_data_opt = self.get_data()
        
        if len(load_data_opt) == 2:
            (self.X_train, self.y_train), (self.X_test, self.y_test) = load_data_opt
        else:
            (self.X_train, self.y_train), (self.X_test, self.y_test), (self.X_sxs_train, self.y_sxs_train), (self.X_sxs_test, self.y_sxs_test), (self.X_sur_train, self.y_sur_train), (self.X_sur_test, self.y_sur_test) = load_data_opt
            
        if self.config.data_loader.scale_data:
            scaler = StandardScaler()
            self.X_train_, self.X_test_ = deepcopy(self.X_train), deepcopy(self.X_test)
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)

        self.X, self.y = np.concatenate([self.X_train, self.X_test]), np.concatenate([self.y_train, self.y_test])
        self.in_out_shapes = {'input_shape' : self.X.shape[1], 'output_shape' : self.y.shape[1]}
        

    def add_latent_data(self, data_tr, data_ts):
        self.latent_train = data_tr
        self.latent_test = data_ts

    def add_umap_data(self, data_tr, data_ts):
        self.umap_train = data_tr
        self.umap_test = data_ts

    def get_autoencoder_train_data(self):
        if self.train_weights is not None:
            return MultiOutputDataGenerator(self.y_train, [self.umap_train, self.y_train], self.config.trainer.batch_size, self.train_weights)
        else:
            return MultiOutputDataGenerator(self.y_train, [self.umap_train, self.y_train], self.config.trainer.batch_size)
    
    def get_autoencoder_test_data(self):
        return MultiOutputDataGenerator(self.y_test, [self.umap_test, self.y_test], self.config.trainer.batch_size)
    
    def get_mapper_train_data(self):
        return DataGenerator(self.umap_train, self.latent_train, self.config.trainer.batch_size)

    def get_mapper_test_data(self):
        return DataGenerator(self.umap_test, self.latent_test, self.config.trainer.batch_size_test)
    
    def get_generator_train_data(self):
        if self.train_weights is not None:
            return DataGenerator(self.X_train, self.y_train, self.config.trainer.batch_size, self.train_weights)
        else:
            return DataGenerator(self.X_test, self.y_test, self.config.trainer.batch_size)
        
    def get_generator_test_data(self):
        return DataGenerator(self.X_test, self.y_test, self.config.trainer.batch_size)
    
    def get_data(self):

        load_data_opt = load_data(self.config.data_loader.data_path, self.config.data_loader.data_output_type)
        split_size = self.config.data_loader.split_size

        if len(load_data_opt)==3:
            opt1, opt2, self.delta_t = load_data_opt
        else:
            opt1, opt2 = load_data_opt

        if len(opt1) == 2:
            x_sxs = opt1[0]
            y_sxs = opt1[1]
            x_sur = opt2[0]
            y_sur = opt2[1]

            (x_sxs_train, y_sxs_train), (x_sxs_test, y_sxs_test) = split_data(x_sxs, y_sxs, split_size)
            (x_sur_train, y_sur_train), (x_sur_test, y_sur_test) = split_data(x_sur, y_sur, split_size)

            x_train, y_train = np.concatenate([x_sur_train, x_sxs_train]), np.concatenate([y_sur_train, y_sxs_train])
            x_test, y_test = np.concatenate([x_sur_test, x_sxs_test]), np.concatenate([y_sur_test, y_sxs_test])

            self.train_weights = np.concatenate([np.ones(len(x_sur_train)), self.config.data_loader.sxs_sample_weight*np.ones(len(x_sxs_train))])

            return (x_train, y_train), (x_test, y_test), (x_sxs_train, y_sxs_train), (x_sxs_test, y_sxs_test), (x_sur_train, y_sur_train), (x_sur_test, y_sur_test)

        else:
            x = opt1
            y = opt2

            return split_data(x, y, split_size)
    
class MappedDataLoader(BaseDataLoader):

    '''
    Data loader class for mapped autoencoder models. Loads data, splits in train and test sets, allows for input scaling and SXS augmented dataset 
    loading. Functions for including the latent space data are implemented. After data loading, calls the necessary data 
    generator classes for each of the fit calls.

    Input parameters:
    -----------------

    config: dict
        Configuration dictionary created from the parsed data of the .json file.
    '''

    def __init__(self, config):
        super(MappedDataLoader, self).__init__(config)

        self.train_weights = None

        load_data_opt = self.get_data()
        
        if len(load_data_opt) == 2:
            (self.X_train, self.y_train), (self.X_test, self.y_test) = load_data_opt
        else:
            (self.X_train, self.y_train), (self.X_test, self.y_test), (self.X_sxs_train, self.y_sxs_train), (self.X_sxs_test, self.y_sxs_test), (self.X_sur_train, self.y_sur_train), (self.X_sur_test, self.y_sur_test) = load_data_opt
            
        if self.config.data_loader.scale_data:
            scaler = StandardScaler()
            self.X_train_, self.X_test_ = deepcopy(self.X_train), deepcopy(self.X_test)
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)

        self.x, self.y = np.concatenate([self.X_train, self.X_test]), np.concatenate([self.y_train, self.y_test])
        self.in_out_shapes = {'input_shape' : self.x.shape[1], 'output_shape' : self.y.shape[1]}

    def add_latent_data(self, data_tr, data_ts):
        self.latent_train = data_tr
        self.latent_test = data_ts

    def get_autoencoder_train_data(self):
        if self.train_weights is not None:
            return DataGenerator(self.y_train, self.y_train, self.config.trainer.batch_size, self.train_weights)
        else:
            return DataGenerator(self.y_train, self.y_train, self.config.trainer.batch_size)
        
    def get_autoencoder_test_data(self):
        return DataGenerator(self.y_test, self.y_test, self.config.trainer.batch_size)
    
    def get_mapper_train_data(self):
        return DataGenerator(self.X_train, self.latent_train, self.config.trainer.batch_size)

    def get_mapper_test_data(self):
        return DataGenerator(self.X_test, self.latent_test, self.config.trainer.batch_size_test)
    
    def get_generator_train_data(self):
        if self.train_weights is not None:
            return DataGenerator(self.X_train, self.y_train, self.config.trainer.batch_size, self.train_weights)
        else:
            return DataGenerator(self.X_test, self.y_test, self.config.trainer.batch_size)
        
    def get_generator_test_data(self):
        return DataGenerator(self.X_test, self.y_test, self.config.trainer.batch_size)
    
    def get_data(self):

        load_data_opt = load_data(self.config.data_loader.data_path, self.config.data_loader.data_output_type)
        split_size = self.config.data_loader.split_size

        if len(load_data_opt)==3:
            opt1, opt2, self.delta_t = load_data_opt
        else:
            opt1, opt2 = load_data_opt

        if len(opt1) == 2:
            x_sxs = opt1[0]
            y_sxs = opt1[1]
            x_sur = opt2[0]
            y_sur = opt2[1]

            (x_sxs_train, y_sxs_train), (x_sxs_test, y_sxs_test) = split_data(x_sxs, y_sxs, split_size)
            (x_sur_train, y_sur_train), (x_sur_test, y_sur_test) = split_data(x_sur, y_sur, split_size)

            x_train, y_train = np.concatenate([x_sur_train, x_sxs_train]), np.concatenate([y_sur_train, y_sxs_train])
            x_test, y_test = np.concatenate([x_sur_test, x_sxs_test]), np.concatenate([y_sur_test, y_sxs_test])

            train_weights = np.concatenate([np.ones(len(x_sur_train)), self.config.data_loader.sxs_sample_weight*np.ones(len(x_sxs_train))])
            self.train_weights = [train_weights, train_weights]

            return (x_train, y_train), (x_test, y_test), (x_sxs_train, y_sxs_train), (x_sxs_test, y_sxs_test), (x_sur_train, y_sur_train), (x_sur_test, y_sur_test)

        else:
            x = opt1
            y = opt2

            return split_data(x, y, split_size)
