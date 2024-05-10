import numpy as np
import tensorflow as tf
import keras
from base.base_model import BaseModel
from keras import layers
from pynndescent import NNDescent
from umap.umap_ import fuzzy_simplicial_set
from sklearn.utils import check_random_state
from umap.umap_ import find_ab_params
from umap.parametric_umap import construct_edge_dataset, umap_loss

class Embedder(BaseModel):
    def __init__(self,config, input_shape, output_shape):
        super(Embedder, self).__init__(config)

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.build_model()

    def build_model(self):
        
        inp = keras.Input(self.input_shape)

        x = layers.Dense(512, activation = "leaky_relu", kernel_initializer = "glorot_uniform")(inp)
        x = layers.Dense(512, activation = "leaky_relu", kernel_initializer = "glorot_uniform")(x)
        x = layers.Dense(512, activation = "leaky_relu", kernel_initializer = "glorot_uniform")(x)
        x = layers.Dense(512, activation = "leaky_relu", kernel_initializer = "glorot_uniform")(x)
        x = layers.Dense(512, activation = "leaky_relu", kernel_initializer = "glorot_uniform")(x)
        x = layers.Dense(512, activation = "leaky_relu", kernel_initializer = "glorot_uniform")(x)

        opt = layers.Dense(self.output_shape)(x)

        self.embedder = keras.Model(inp, opt)

class ParametricUMAP(BaseModel):
    def __init__(self, config, data_loader):
        super(ParametricUMAP, self).__init__(config)

        data_reg_tr = data_loader.X_train
        n_neighbors = config.parametric_umap.n_neighbors
        metric = config.parametric_umap.metric
        edge_dataset_n_epochs = config.parametric_umap.edge_dataset_n_epochs
        edge_dataset_batch_size = config.parametric_umap.edge_dataset_batch_size
        min_dist = config.parametric_umap.min_dist
        self.epochs = config.paramteric_umap.epochs
        self.input_shape_reg = data_loader.in_out_shapes["input_shape"]

        self.embedder = Embedder(config, self.input_shape_reg, config.model.latent_dim).embedder

        n_trees = 5 + int(round((data_reg_tr.shape[0]) ** 0.5 / 20.0))

        # max number of nearest neighbor iters to perform
        n_iters = max(5, int(round(np.log2(data_reg_tr.shape[0]))))

        negative_sample_rate = 5 

        print("\n\nComputing nearest neighbors for parametric UMAP\n\n")

        # get nearest neighbors
        nnd = NNDescent(
            data_reg_tr,
            n_neighbors=n_neighbors,
            metric=metric,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            verbose=True
        )
        # get indices and distances
        knn_indices, knn_dists = nnd.neighbor_graph
        random_state = check_random_state(None)

        umap_graph, sigmas, rhos = fuzzy_simplicial_set(
            X = data_reg_tr,
            n_neighbors = n_neighbors,
            metric = metric,
            random_state = random_state,
            knn_indices= knn_indices,
            knn_dists = knn_dists,
        )
        (
            self.edge_dataset,
            self.batch_size,
            self.n_edges,
            head,
            tail,
            edge_weight,
        ) = construct_edge_dataset(
            data_reg_tr,
            umap_graph,
            n_epochs = edge_dataset_n_epochs,
            batch_size = edge_dataset_batch_size,
            parametric_embedding = True,
            parametric_reconstruction = False,
            global_correlation_loss_weight = 0
        )

        _a, _b = find_ab_params(1.0, min_dist)

        self.umap_loss_fn = umap_loss(
            self.batch_size,
            negative_sample_rate,
            _a,
            _b,
            edge_weight,
            parametric_embedding = True
        )

        self.build_model()

    def build_model(self):
        # define the inputs
        to_x = tf.keras.layers.Input(shape=self.input_shape_reg, name="to_x")
        from_x = tf.keras.layers.Input(shape=self.input_shape_reg, name="from_x")
        inputs = [to_x, from_x]

        # parametric embedding
        embedding_to = self.embedder(to_x)
        embedding_from = self.embedder(from_x)

        # concatenate to/from projections for loss computation
        embedding_to_from = tf.concat([embedding_to, embedding_from], axis=1)
        embedding_to_from = tf.keras.layers.Lambda(lambda x: x, name="umap")(embedding_to_from)

        outputs = {'umap': embedding_to_from}

        # create model
        self.parametric_model = tf.keras.Model(inputs=inputs, outputs=outputs,)

        self.parametric_model.compile(loss=self.umap_loss_fn, optimizer = "adam")

        self.steps_per_epoch = int(
            self.n_edges / self.batch_size / 5
        )

