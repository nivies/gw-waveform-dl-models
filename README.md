# Gravitational waveform DL based modelling

Master's thesis project for developing a neural network based gravitational waveform model. Based off Ahmkel's [Keras-Project-Template](https://github.com/Ahmkel/Keras-Project-Template).

# Table of contents

- [Scripts](#scripts)
- [Configuration file structure](#config-file-structure)
- [Project architecture](#project-architecture)
    - [Configs](#configs)
    - [Experiment logs](#experiment_logs)
    - [Experiments](#experiments)
    - [Figures](#figures)
    - [Base](#base)
    - [Data loader](#data_loader)
    - [Models](#models)
    - [Trainers](#trainers)
    - [Utils](#utils)

<a name="scripts"></a>
# Scripts 

All the presented results can be reproduced from this project. The scripts in the root folder are:

- <h3>Training script

Script's name is `main.py`. It parses all the information in the config.json file then declares and trains the model. Usage:

```
python main.py -c [path to configuration file]
```


- <h3>Evaluation script

Script's name is `evaluate.py`. It loads a trained model and evaluates on the training and testing data. It then plots a mismatch histogram and four examples of a waveform generated using the loaded model along with the ground truth. The best and worse case scenarios along with the case with a mismatch in the 50th percentile of the mismatch distribution and the case in the 10th percentile are plotted.

```
usage: evaluate.py [-h] [-d D] [-l LC]

options:
  -h, --help            show this help message and exit
  -d D, --output_directory D
                        Output directory for the plots
  -l LC, --load_checkpoint LC
                        Directory of checkpoint to load model
```

- <h3>Timing script

Script's name is `timing_script.py`. It loads trained model and uses python's `timeit` library to time the generation time for each model (and surrogate models). 

```
usage: timing_script.py [-h] [-gm] [-pu] [-bs] [-d] [-n] [-en]

Script for timing generation times from different algorithms.

options:
  -h, --help            show this help message and exit
  -gm , --generation_method 
                        Method for GW generation: dense, mapped, regularized, cvae or surrogate.
  -pu , --processing_unit 
                        Whether to run the generation method on cpu ('cpu') or on gpu ('gpu').
  -bs , --batch_size    Batch size for predict method for NN based generation methods.
  -d , --dataset        Dataset to run the script on. 1 -> Mass BNS 2 -> Mass-lambda BNS 3 -> Mass BBH 4 -> Mass z-spin BBH 5 -> Mass full-spin BBH
  -n , --n_datapoints   Number of waveforms to generate.
  -en , --execution_number 
                        Number of script executions for timing.
```


<a name="config-file-structure"></a>
# Config file structure 
The config file contains all necessary information for training a model and loading it afterwards. This section shows an example for the regularized autoencoder. Configuration files for the other architectures follow the same structure, but with omitted elements. The file is structured as follows:

```
├── exp                 
│     └── name          - Name of the run's folder. This folder will be created inside a directory with the date and time of the training.
│
├── data_loader
│     ├── name          - Name of the data_loader class 
│     ├── args          - Arguments to pass to the data_loader class.
│     ├── data_path     - Path to the hdf5 file containing the dataset. The hdf5 file must be organized with two datasets named "waveforms" and "parameters". 
│     ├── scale_data    - Whether to use sklearn's StandardScaler on the input parameters or not.
│     ├── data_output_type
│     │                 - Form of the waveforms to fit. It can be "hp" for the plus polarization, "hc" for the cross polarization or "amplitude_phase" for
│     │                   concatenating the amplitude profile and an unwrapped version of the phase.
│     ├── split_size    - Portion of the dataset to be reserved as test set.
│     └── sxs_sample_weight
│                       - In case of loading the SXS augmented dataset, sample weight applied to the SXS datapoints.
│
├── model 
│     ├── name          - Name of the desired model class.
│     ├── args          - Arguments to pass to the model class.
│     ├── latent_dim    - Latent dimensionality for the autoencoder.  
│     ├── reg_weight    - Value for the lambda in the regularized autoencoder's loss function. Controls the tradeoff between latent regularization 
│                         and reconstruction error.
│     ├── mapper_architecture
│     │                 - Architecture to be used for mapping the UMAP embedding to the latent space. It can be "dense" or "convolutional". Residual connections 
│     │                   used in both architectures
│     ├── mapper_dropout
│     │                 - Dropout probability for the mapper architecture. Only applies to "convolutional" architecture.
│     └── optimizer_kwargs
│                       - Arguments for the optimizer. By default, the used optimizer is Keras' implementation of Adam.
├── parametric_umap
│     ├── n_neighbors   - Value for the number of neighbors to take into account when computing the nearest neighbors graph for the UMAP projection.
│     ├── metric        - Metric type for the nearest neighbors algorithm.
│     ├── min_dist      - Value for the min_dist UMAPS' parameter.
│     ├── steps_per_epoch
│     │                 - Number of steps per epoch in training process.
│     ├── edge_dataset_batch_size  
│     │                 - Batch size for the edges dataset.
│     ├── edge_dataset_n_epochs  
│     │                 - Number of epochs for the edge dataset training. 
│     └── epochs        - Number of epochs for the embedder network training.
│
├── trainer
│     ├── name          - Name of the desired trainer class.
│     ├── args          - Arguments for the trainer class.
│     ├── num_epochs    - Number of epochs for the autoencoder/dense model.
│     ├── mapper_epochs - Number of epochs for the mapper class.
│     ├── batch_size    - Training batch size.
│     ├── batch_size_test
│     │                 - Batch size for validation split.
│     └── verbose_training
│                       - Value for the verbose argument of keras' fit method.
└── callbacks
      ├── checkpoint_monitor
      │                 - "loss" or "val_loss". Metric to apply checkpoint saving.
      ├── checkpoint_mode
      │                 - "min" if the loss has to be minimized or "max" if it has to be maximized.
      ├── checkpoint_save_best_only
      │                 - bool. Whether to save every time or only if loss or val_loss improves.
      ├── checkpoint_save_weights_only
      │                 - bool. Whether to save only the weights or the whole keras model.
      ├── checkpoint_verbose
      │                 - bool. Whether to print information about the callback activation or not.
      ├── tensorboard_write_graph
      │                 - bool. Whether to write the execution graph to the tensorboard callback or not.
      ├── early_stopping_patience
      │                 - Number of epochs without improvement tolerated before terminating the training process.
      ├── lr_reduce_factor
      │                 - Multiplicative factor to reduce the learning rate when plateauing.
      ├── lr_reduce_patience
      │                 - Number of epochs without improvement tolerated before the learning rate is reduced.
      └── min_lr        - Inferior threshold for the learning rate.
```

<a name="project-architecture"></a>
# Project architecture

The project code is organized according to its function in different directories. The root directory is reserved for scripts that use the utilities defined in each folder.

<a name="data"></a>
## data

Directory for the hdf5 data files.

<a name="configs"></a>
## configs

The configs directory holds the .json configuration files.

<a name="experiment_logs"></a>
## experiment_logs

Directory for the training logs.

<a name="experiments"></a>
## experiments

Directory for the saved models. Every model is saved in a folder with the datetime and hour when the training starts. In this directory, another folder is created with the name specified in the configuration file. This folder contains two folders:

1. A checkpoint folder with the weights for every best model trained, (joblib's) serialized files with the training losses for each training performed and the particular configuration file used for the experiment.
2. A tensorboard directory for the usage of the Tensorboard callback.

<a name="figures"></a>
## figures

Directory containing relevant figures and mismatch files for particular models. The directory name references the model used and the dataset in which it was trained.

- dense/mapper/reg reference a dense NN, the mapped autoencoder and the regularized autoencoder.
- q/qz/qzp reference BBH datasets in which the mass ratio is used (q), mass ratio + orbit-aligned spin (qz) and mass ratio + randomly aligned spin (qzp).
- q/q_lbd reference BNS datasets with mass ratio (q) or mass ratio + tidal deformabilities (q_lbd).

In each folder, a serialized (using joblib) binary file with train and test set mismatches is stored. Two sub-directories with train and test figures are also in this folder. Each contains the corresponding mismatch histogram and four waveform plots in which a comparison between the generated waveform and the ground truth are compared. Each of the four plots correspond to the worst and best case scenarios, the waveform corresponding to the mismatch at the 50th percentile of the mismatch distribution and the one corresponding to the 10th percentile.

<a name="base"></a>
## base

Directory containing abstract definitions for the data loader, trainer and model classes. Each of the defined classes for these purposes are inherited from the base directory.

<a name="data_loader"></a>
## data_loader

Directory containing the data loader classes. It has code for data generation for the training loop, logic for the data loading and some preprocessing utilities, such as standard scaling,, shuffling and train-test splits.

<a name="models"></a>
## models

Directory containing the declaration of all the used neural networks and the overall architectures for each proposed model.

<a name="trainers"></a>
## trainers

Directory containing the classes where all the callbacks are defined and the fit methods are called according to the configuration file.

<a name="utils"></a>
## utils

Directory containing a set of functions to be called at any point of the project's pipeline.

### args

Functions for the parsing of the script's arguments.

### config

Functions for parsing the configuration file, loading the Checkpoint callbacks and initializing instances of the corresponding classes from the information of the configuration files.

### data_preprocessing

Functions for loading data from a particularly formatted .hdf5 file, for the SXS dataset and for some preprocessing utilities: treatment of the waveform phases (setting them to 0 and unwrapping them), correct formatting according to the configuration file (polarization selection or amplitude-phase representation).

### dirs

Directory creation utilities.

### eval

Model and data loading utilities. Supports inference mode for quick loading of regularized autoencoders (bypasses the parametric UMAP declaration as it doesn't have to be trained and instead, directly loads the trained network).

### plot_utils

Plotting utilities. There is code for plotting the waveform comparisons, the histogram, several histograms in order to compare them and automatizating functions.

### utils

Several miscellaneous.
