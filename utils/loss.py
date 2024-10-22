import tensorflow as tf
import keras.backend as K
from keras.losses import mean_absolute_error
from tensorflow.keras.callbacks import Callback
from keras.regularizers import Regularizer

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
    
    return  K.abs(1. - overl)

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
    
    return  K.abs(1. - overl)



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