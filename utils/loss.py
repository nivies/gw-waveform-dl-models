import tensorflow as tf
import keras.backend as K
from keras.losses import mean_absolute_error
from tensorflow.keras.callbacks import Callback
from keras.regularizers import Regularizer
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import os

class GradientDiagnosticsCallback(tf.keras.callbacks.Callback):
    def __init__(self, config, data_loader, folder_name, batch_size = 32, threshold=1e-6, plot_interval=5, autoencoder = True):
        """
        Initialize the callback.
        :param model: The model to analyze.
        :param x_train: Training inputs as a NumPy array.
        :param y_train: Training targets as a NumPy array.
        :param batch_size: Batch size for gradient computation.
        :param threshold: Threshold for flagging vanishing gradients.
        :param log_file: Path to a file for logging gradient diagnostics. If None, logs to console only.
        :param plot_interval: Interval (in epochs) to generate gradient distribution plots.
        """
        super().__init__()
        self.data_loader = deepcopy(data_loader)
        self.threshold = threshold
        self.batch_size = batch_size
        self.plot_interval = plot_interval
        self.epoch_gradient_norms = []  # To store gradient norms for plotting

        if autoencoder:
            self.data_loader.X_train = self.data_loader.y_train
            self.data_loader.X_test = self.data_loader.y_test

        root_dir = os.path.dirname(config.callbacks.checkpoint_dir)
        self.log_dir = os.path.join(os.path.dirname(root_dir), folder_name)
        self.log_file = os.path.join(self.log_dir, "gradient_control.txt")


        if self.log_file:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            # with open(log_file, "w") as f:
            #     f.write("Epoch,Mean,Max,Min,StdDev,FlaggedLayers\n")

    def on_epoch_end(self, epoch, logs=None):

        x_train = self.data_loader.X_train
        y_train = self.data_loader.y_train
        indices = np.random.choice(len(x_train), self.batch_size, replace=False)
        x_batch = x_train[indices]
        y_batch = y_train[indices]

        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self.model(x_batch, training=True)
            if len(predictions) == 2:
                predictions = predictions[1]
            loss = self.model.compiled_loss(tf.convert_to_tensor(y_batch, dtype=tf.float32), predictions)

        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Get layer names corresponding to the gradients
        layer_names = [var.name for var in self.model.trainable_variables]

        # Calculate gradient norms
        gradient_norms = [tf.norm(grad).numpy() for grad in gradients if grad is not None]
        self.epoch_gradient_norms.append(gradient_norms)

        # Summary statistics
        mean_norm = np.mean(gradient_norms)
        max_norm = np.max(gradient_norms)
        min_norm = np.min(gradient_norms)
        std_norm = np.std(gradient_norms)

        # Identify flagged layers by name
        flagged_layers = [(name, norm) for name, norm in zip(layer_names, gradient_norms) if norm < self.threshold]

        # Log diagnostics
        log_message = (f"Epoch {epoch + 1}: "
                    f"Mean: {mean_norm:.4e}, Max: {max_norm:.4e}, Min: {min_norm:.4e}, StdDev: {std_norm:.4e}, "
                    f"Nº of flagged layers: {len(flagged_layers)} | "
                    f"Flagged Layers: {[name for name, _ in flagged_layers]}\n")

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(log_message)
        else:
            print(log_message)


        # Plot gradient distribution every `plot_interval` epochs
        if (epoch + 1) % self.plot_interval == 0:
            self._plot_gradient_distribution(epoch + 1)

    def _plot_gradient_distribution(self, epoch):
        # Flatten all gradient norms for the epoch
        all_norms = np.concatenate(self.epoch_gradient_norms)

        # Filter out zero gradients to avoid issues with log scale
        all_norms = all_norms[all_norms > 0]

        plt.figure(figsize=(10, 6))
        plt.hist(all_norms, bins=30, alpha=0.7)  # Logarithmic scale for frequency
        plt.title(f"Log-Scale Gradient Norm Distribution at Epoch {epoch}")
        plt.xscale('log')
        plt.xlabel("Gradient Norm (Log Scale)")
        plt.ylabel("Number of layers")
        plt.grid(True)

        # Save or display the plot
        plot_path = os.path.join(self.log_dir, f"gradient_distribution_epoch_{epoch}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Gradient distribution plot saved to {plot_path}")


def mean_absolute_error_batched(y_true, y_pred, batch_size):
    """
    Computes the Mean Absolute Error (MAE) in batches.
    
    Args:
        y_true: Ground truth values (tensor).
        y_pred: Predicted values (tensor).
        batch_size: Size of each batch.
        
    Returns:
        Mean Absolute Error computed over the entire dataset.
    """
    # Create TensorFlow dataset and batch it
    dataset = tf.data.Dataset.from_tensor_slices((y_true, y_pred)).batch(batch_size)
    
    # Initialize the total MAE and number of batches
    total_mae = 0
    num_batches = 0
    
    # Iterate over each batch and compute MAE for that batch
    for batch_y_true, batch_y_pred in dataset:
        batch_mae = tf.reduce_mean(tf.abs(batch_y_pred - batch_y_true))
        total_mae += batch_mae
        num_batches += 1
    
    # Compute the overall MAE by averaging across batches
    overall_mae = total_mae / num_batches
    
    return overall_mae

def overlap_hphc_batched(h1, h2, batch_size, dt=2*4.925794970773135e-06):
    dataset = tf.data.Dataset.from_tensor_slices((h1, h2)).batch(batch_size)
    
    results = []
    
    for h1_batch, h2_batch in dataset:
        split_size = int(4096 / 2)

        # Split the real (hp) and imaginary (hc) parts
        h1_hp, h1_hc = h1_batch[:, :split_size], h1_batch[:, split_size:]
        h2_hp, h2_hc = h2_batch[:, :split_size], h2_batch[:, split_size:]
        
        # Reconstruct h1 and h2 as complex numbers
        h1_complex = tf.cast(h1_hp, tf.complex64) + 1j * tf.cast(h1_hc, tf.complex64)
        h2_complex = tf.cast(h2_hp, tf.complex64) + 1j * tf.cast(h2_hc, tf.complex64)
        
        # Perform FFT
        h1_f = tf.signal.fft(h1_complex) * dt
        h2_f = tf.signal.fft(h2_complex) * dt
        
        df = 1.0 / 2048 / dt
        sig_norm = 4 * df
        
        sig1 = K.sqrt(tf.cast(tf.math.reduce_sum(tf.math.conj(h1_f) * h1_f, axis=-1), tf.float32) * sig_norm)
        sig2 = K.sqrt(tf.cast(tf.math.reduce_sum(tf.math.conj(h2_f) * h2_f, axis=-1), tf.float32) * sig_norm)
        
        norm = 1 / sig1 / sig2
        inner = tf.cast(tf.math.reduce_sum(tf.math.conj(h1_f) * h2_f, axis=-1), tf.float32)
        overl = tf.cast(4 * df * inner * norm, tf.float32)
        
        results.append(K.abs(1. - overl))
    
    return tf.concat(results, axis=0)

def overlap_amp_phs_batched(h1, h2, batch_size, dt=2*4.925794970773135e-06):
    # Create a TensorFlow dataset from the input tensors h1 and h2
    dataset = tf.data.Dataset.from_tensor_slices((h1, h2)).batch(batch_size)
    
    results = []
    
    for h1_batch, h2_batch in dataset:
        # Perform the computations on the current batch
        
        split_size = int(4096 / 2)
        
        # Split the amplitude and phase
        h1_amp, h1_phs = h1_batch[:, :split_size], h1_batch[:, split_size:]
        h2_amp, h2_phs = h2_batch[:, :split_size], h2_batch[:, split_size:]
        
        # Reconstruct h1 and h2 as complex numbers
        h1_complex = tf.cast(h1_amp, tf.complex64) * tf.math.exp(1j * tf.cast(h1_phs, tf.complex64))
        h2_complex = tf.cast(h2_amp, tf.complex64) * tf.math.exp(1j * tf.cast(h2_phs, tf.complex64))
        
        # Perform FFT on both h1 and h2
        h1_f = tf.signal.fft(h1_complex) * dt
        h2_f = tf.signal.fft(h2_complex) * dt
        
        df = 1.0 / 2048 / dt
        sig_norm = 4 * df
        
        # Compute the norms of h1 and h2
        sig1 = K.sqrt(tf.cast(tf.math.reduce_sum(tf.math.conj(h1_f) * h1_f, axis=-1), tf.float32) * sig_norm)
        sig2 = K.sqrt(tf.cast(tf.math.reduce_sum(tf.math.conj(h2_f) * h2_f, axis=-1), tf.float32) * sig_norm)
        
        # Compute the inner product and overlap
        norm = 1 / sig1 / sig2
        inner = tf.cast(tf.math.reduce_sum(tf.math.conj(h1_f) * h2_f, axis=-1), tf.float32)
        overl = tf.cast(4 * df * inner * norm, tf.float32)
        
        # Append the batch result to the results list
        results.append(K.abs(1. - overl))
    
    # Concatenate all batch results into a single tensor
    return tf.concat(results, axis=0)

class DynamicLossWeightsCallback(Callback):
    def __init__(self, initial_weights, final_weights, begin_transition_epoch, end_transition_epoch):
        """
        Initialize the callback.
        Args:
        - initial_weights: Initial loss weights for the outputs at the start of training.
        - final_weights: Final loss weights for the outputs at the end of training.
        - weight_transition_epoch: Epoch after which the weights should start transitioning.
        """
        super().__init__()
        self.initial_weights = initial_weights
        self.final_weights = final_weights
        self.begin_transition_epoch = begin_transition_epoch
        self.end_transition_epoch = end_transition_epoch

    def on_epoch_begin(self, epoch, logs=None):
        """Update the loss weights dynamically at the start of each epoch."""
        if epoch < self.begin_transition_epoch:
            # Use initial weights before the transition epoch
            loss_weights = self.initial_weights
        else:
            # Gradually transition to the final weights
            alpha = min(1.0, (epoch - self.begin_transition_epoch) / (self.end_transition_epoch - self.begin_transition_epoch))
            loss_weights = [
                (1 - alpha) * init_w + alpha * final_w
                for init_w, final_w in zip(self.initial_weights, self.final_weights)
            ]
            print(f"\nEpoch {epoch+1}: Updated loss weights to {loss_weights}")
        
        # Update model's loss weights
        self.model.compiled_loss._loss_weights = loss_weights

def overlap_amp_phs(h1, h2, dt=2*4.925794970773135e-06, df=None):
    
    split_size = int(4096/2)

    h1_amp, h1_phs = h1[:, :split_size], h1[:, split_size:]
    h2_amp, h2_phs = h2[:, :split_size], h2[:, split_size:]

    h1 =  tf.cast(h1_amp, tf.complex64)*tf.math.exp(1j*(tf.cast(h1_phs, tf.complex64)))
    h2 = tf.cast(h2_amp, tf.complex64)*tf.math.exp(1j*(tf.cast(h2_phs, tf.complex64)))

    h1_f = tf.signal.fft(h1)*dt
    h2_f = tf.signal.fft(h2)*dt
    
    df = 1.0 /  2048 / dt
    sig_norm = 4*df

    sig1 = K.sqrt(tf.cast((tf.math.reduce_sum(tf.math.conj(h1_f)*h1_f,axis=-1)),tf.float32)*sig_norm)
    sig2 = K.sqrt(tf.cast((tf.math.reduce_sum(tf.math.conj(h2_f)*h2_f,axis=-1)),tf.float32)*sig_norm)
    
    norm = 1/sig1/sig2
    inner = tf.cast(tf.math.reduce_sum((tf.math.conj(h1_f)*h2_f),axis=-1),tf.float32)
    overl = tf.cast((4*df*inner*norm),tf.float32)
    
    return  tf.math.log(1. - overl) / tf.math.log(10.0)

def overlap_hphc(h1, h2, dt=2*4.925794970773135e-06, df=None):
    
    split_size = int(4096/2)

    h1_hp, h1_hc = h1[:, :split_size], h1[:, split_size:]
    h2_hp, h2_hc = h2[:, :split_size], h2[:, split_size:]

    h1 =  tf.cast(h1_hp, tf.complex64) + 1j*(tf.cast(h1_hc, tf.complex64))
    h2 =  tf.cast(h2_hp, tf.complex64) + 1j*(tf.cast(h2_hc, tf.complex64))

    h1_f = tf.signal.fft(h1)*dt
    h2_f = tf.signal.fft(h2)*dt
    
    df = 1.0 /  2048 / dt
    sig_norm = 4*df

    sig1 = K.sqrt(tf.cast((tf.math.reduce_sum(tf.math.conj(h1_f)*h1_f,axis=-1)),tf.float32)*sig_norm)
    sig2 = K.sqrt(tf.cast((tf.math.reduce_sum(tf.math.conj(h2_f)*h2_f,axis=-1)),tf.float32)*sig_norm)
    
    norm = 1/sig1/sig2
    inner = tf.cast(tf.math.reduce_sum((tf.math.conj(h1_f)*h2_f),axis=-1),tf.float32)
    overl = tf.cast((4*df*inner*norm),tf.float32)
    
    # return  K.abs(1. - overl)
    return  tf.math.log(1. - overl) / tf.math.log(10.0)



def ovlp_mae_loss_amp_phs(y_pred, y_true):
    
    return overlap_amp_phs(y_pred, y_true) + mean_absolute_error(y_pred, y_true)


def ovlp_mae_loss_hphc(y_pred, y_true):
    
    return overlap_hphc(y_pred, y_true) + mean_absolute_error(y_pred, y_true)

class ComponentWiseRegularizer(Regularizer):
    def __init__(self, coef=1e-7, exp=0.5):
        self.coef = coef
        self.exp = exp

    def __call__(self, x):
        # Ensure that the regularizer returns a scalar by reducing the result
        reg_vector = self.get_regularization_vector(x.shape[-1])
        regularization = tf.abs(x) * reg_vector  # Squaring x for L2-like regularization
        return tf.reduce_sum(regularization)

    def get_regularization_vector(self, vector_length):
        # Create a vector for the regularization term
        x = tf.range(0, vector_length, dtype = tf.float32)
        return tf.expand_dims((x * self.coef) ** self.exp, axis = 0) # axis changed from -1 to 0

    def get_config(self):
        # Return the configuration as required for a Keras custom object
        return {'coef': float(self.coef), 'exp': float(self.exp)}