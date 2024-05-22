
from base.base_model import BaseModel
import keras
from keras import layers

class ConvBlock1D_test(BaseModel):

    '''
    Convolutional block model instance.

    Defines a 1D convolutional block consisting in 3 repetitions of:

    1. 1D convolutional layer.
    2. 1D Spatial Dropout.
    3. 1D Upsampling layer.
    4. Leaky ReLU activation.

    Input and output dimensions are set equal in order to implement a residual network with Add layers. This block is set
    to be called by the UMAP mapper class if the argument model.mapper_architecture is set to convolutional.

    Input arguments:
    ----------------
    config: dict
        Config dict loaded from config .json file.

    input_shape: int 
        Number of input and output dimensions of convolutional block.
    ----------------
    Declared model is stored in the .conv_block method.

    '''

    def __init__(self, config, input_shape):
        super(ConvBlock1D_test, self).__init__(config)

        self.input_shape = input_shape
        self.build_model()
    
    def build_model(self):

        dropout_rate = self.config.model.mapper_dropout

        inp = keras.Input(self.input_shape)
        x = layers.Reshape((self.input_shape, 1))(inp)

        x = layers.Conv1D(filters = 512, kernel_size = 3, padding = "same")(x)
        x = layers.SpatialDropout1D(rate = dropout_rate)(x)
        x = layers.UpSampling1D(size = 2)(x)
        x = layers.LeakyReLU()(x)
                
        x = layers.Conv1D(filters = 512, kernel_size = 3, padding = "same")(x)
        x = layers.SpatialDropout1D(rate = dropout_rate)(x)
        x = layers.UpSampling1D(size = 2)(x)
        x = layers.LeakyReLU()(x)

                
        x = layers.Conv1D(filters = 512, kernel_size = 3, padding = "same")(x)
        x = layers.SpatialDropout1D(rate = dropout_rate)(x)
        x = layers.UpSampling1D(size = 2)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Flatten()(x)

        x = layers.Dense(self.input_shape)(x)

        self.conv_block = keras.Model(inp, x)

class RegularizedAutoEncoder_test(BaseModel):

    '''
    Class for the definition of the 1D convolutional autoencoder architecture with a regularization in the latent space. Called from the 
    RegularizedAutoEncoderGenerator class. It defines a preset autoencoder architecture with the necessary logic for the 
    regularization implementation.

    Input arguments:
    ----------------

    config: dict
        Dict built from the configuration .json file.

    out_shape: int
        Parameter controlling the dimensionality of the data to fit. Particularly, the size of the waveform vectors.
    ----------------

    Autoencoder built stored in .autoencoder method. Similarly, the encoder and decoder are stored in the .encoder and the
    .decoder methods.
    '''


    def __init__(self, config, out_shape):
        super(RegularizedAutoEncoder_test, self).__init__(config)
        self.latent_dim = config.model.latent_dim
        self.out_shape = out_shape
        self.build_model()

    def build_model(self):
        
        inp = keras.Input(self.out_shape)

        x = layers.Dense(units=1024, activation='leaky_relu', kernel_initializer='glorot_uniform')(inp)
        x = layers.Dense(units=512, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)
        x = layers.Dense(units=512, activation='leaky_relu', kernel_initializer='glorot_uniform')(x)

        enc = layers.Dense(units=self.latent_dim, name = 'encoding')(x)

        y = layers.Dense(units=512, activation='leaky_relu', kernel_initializer='glorot_uniform')(enc)
        y = layers.Dense(units=1024, activation='leaky_relu', kernel_initializer='glorot_uniform')(y)
        y = layers.Dense(units=1024, activation='leaky_relu', kernel_initializer='glorot_uniform')(y)
        
        dec = layers.Dense(units=self.out_shape, activation='linear', name = 'reconstruction')(y)

        self.autoencoder = keras.Model(inp, [enc, dec])
        self.encoder = keras.Model(inp, enc)
        self.decoder = keras.Model(enc, dec)

        optimizer = keras.optimizers.Adam(**self.config.model.optimizer_kwargs)
        self.autoencoder.compile(optimizer = optimizer,
                                loss_weights=[self.config.model.reg_weight, 1-self.config.model.reg_weight],
                                loss = {'encoding' : 'mean_absolute_error', 'reconstruction' : 'mean_absolute_error'}
                                )
        

class Embedder_test(BaseModel):

    '''
    Class for declaring the embedding network that creates the UMAP embedding. This class is called either from the 
    Parametric UMAP class, or from the RegularizedAutoEncoderGenerator directly if the inference parameter is set to True.

    Input parameters:
    -----------------

    config: dict
        Configuration dictionary built from the configuration .json file.

    input_shape: int
        Input parameter shape.

    output_shape: int
        Embedding dimensionality (which is set to match the latent space dimensionality of the autoencoder).
    -----------------

    Declared network is stored in the .embedder method.
    '''

    def __init__(self, config, input_shape, output_shape):
        super(Embedder_test, self).__init__(config)

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.build_model()

    def build_model(self):
        
        inp = keras.Input(self.input_shape)

        x = layers.Dense(512, activation = "leaky_relu", kernel_initializer = "glorot_uniform")(inp)

        opt = layers.Dense(self.output_shape)(x)

        self.embedder = keras.Model(inp, opt)

class UMAPMapper_test(BaseModel):

    '''
    Regularized autoencoder mapper from the UMAP embedding to the autoencoder's latent space. Built entirely from 
    .json configuration file. Declares a the model using Keras' functional api with the appropriate input and output
    dimensions. If model.mapper_architecture field from configuration .json file is set to "dense", it defines a dense-based
    architecture with concatenations of the input from the UMAP embedding. If, on the contrary, it is set to "convolutional" it
    makes use of the ConvBlock1D class to build a residual network.

    Input arguments:
    ----------------
    config: dict 
        Configuration dictionary generated from loading the configuration .json file.
    ----------------
    Declared model is in the .mapper method.
    '''

    def __init__(self, config):
        super(UMAPMapper_test, self).__init__(config)

        if self.config.model.mapper_architecture == "dense":
            self.build_model_dense()
        elif self.config.model.mapper_architecture == "convolutional":
            self.build_model_convolutional()
        else:
            raise NameError("Mapper architecture not implemented!")

    def build_model_dense(self):

        input_output_shape = self.config.model.latent_dim

        inp = keras.Input(input_output_shape)

        x = layers.Dense(512, activation='leaky_relu', kernel_initializer='he_uniform')(inp)
        x = layers.Dense(512, activation='leaky_relu', kernel_initializer='he_uniform')(x)
        x = layers.Dense(512, activation='leaky_relu', kernel_initializer='he_uniform')(x)

        opt = layers.Dense(input_output_shape)(x)

        self.mapper = keras.Model(inp, opt)

        optimizer = keras.optimizers.Adam(**self.config.model.optimizer_kwargs)

        self.mapper.compile(optimizer = optimizer,
                           loss = 'mae'
                           )

    def build_model_convolutional(self):
        
        input_output_shape = self.config.model.latent_dim

        inp = keras.Input(input_output_shape)

        y1 = ConvBlock1D_test(self.config, input_output_shape).conv_block(inp)
        x1 = layers.Add()([inp, y1])

        y2 = ConvBlock1D_test(self.config, input_output_shape).conv_block(x1)
        x2 = layers.Add()([inp, y1, y2])

        y3 = ConvBlock1D_test(self.config, input_output_shape).conv_block(x2)
        opt = layers.Add()([inp, y1, y2, y3])

        self.mapper = keras.Model(inp, opt)

        optimizer = keras.optimizers.Adam(**self.config.model.optimizer_kwargs)

        self.mapper.compile(optimizer = optimizer,
                           loss = 'mae'
                           )
        
