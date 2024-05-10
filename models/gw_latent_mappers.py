from base.base_model import BaseModel
import keras
from keras import layers

class ConvBlock1D(BaseModel):
    def __init__(self, config, input_shape):
        super(ConvBlock1D, self).__init__(config)

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

class UMAPMapper(BaseModel):
    def __init__(self, config):
        super(UMAPMapper, self).__init__(config)

        if self.config.model.mapper_architecture == "dense":
            self.build_model_dense()
        elif self.config.model.mapper_architecture == "convolutional":
            self.build_model_convolutional()
        else:
            raise NameError("Mapper architecture not implemented!")

    def build_model_dense(self):

        input_output_shape = self.config.model.latent_dim

        inp = keras.Input(input_output_shape)

        x = layers.Dense(1024, activation='leaky_relu', kernel_initializer='he_uniform')(inp)
        x = layers.Dense(1024, activation='leaky_relu', kernel_initializer='he_uniform')(x)
        x = layers.Concatenate()([inp, x])
        x = layers.Dense(1024, activation='leaky_relu', kernel_initializer='he_uniform')(x)
        x = layers.Dense(1024, activation='leaky_relu', kernel_initializer='he_uniform')(x)
        x = layers.Concatenate()([inp, x])
        x = layers.Dense(1024, activation='leaky_relu', kernel_initializer='he_uniform')(x)
        x = layers.Dense(1024, activation='leaky_relu', kernel_initializer='he_uniform')(x)
        x = layers.Concatenate()([inp, x])
        x = layers.Dense(1024, activation='leaky_relu', kernel_initializer='he_uniform')(x)
        x = layers.Dense(1024, activation='leaky_relu', kernel_initializer='he_uniform')(x)

        opt = layers.Dense(input_output_shape)(x)

        self.mapper = keras.Model(inp, opt)

        optimizer = keras.optimizers.Adam(**self.config.model.optimizer_kwargs)

        self.mapper.compile(optimizer = optimizer,
                           loss = 'mae'
                           )

    def build_model_convolutional(self):
        
        input_output_shape = self.config.model.latent_dim

        inp = keras.Input(input_output_shape)

        y1 = ConvBlock1D(self.config, input_output_shape).conv_block(inp)
        x1 = layers.Add()([inp, y1])

        y2 = ConvBlock1D(self.config, input_output_shape).conv_block(x1)
        x2 = layers.Add()([inp, y1, y2])

        y3 = ConvBlock1D(self.config, input_output_shape).conv_block(x2)
        x3 = layers.Add()([inp, y1, y2, y3])

        y4 = ConvBlock1D(self.config, input_output_shape).conv_block(x3)
        x4 = layers.Add()([inp, y1, y2, y3, y4])

        y5 = ConvBlock1D(self.config, input_output_shape).conv_block(x4)
        x5 = layers.Add()([inp, y1, y2, y3, y4, y5])
        
        y6 = ConvBlock1D(self.config, input_output_shape).conv_block(x5)
        opt = layers.Add()([inp, y1, y2, y3, y4, y5, y6])

        self.mapper = keras.Model(inp, opt)

        optimizer = keras.optimizers.Adam(**self.config.model.optimizer_kwargs)

        self.mapper.compile(optimizer = optimizer,
                           loss = 'mae'
                           )
        
