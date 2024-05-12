# Gravitational waveform DL based modelling

Master's thesis project for developing a neural network based gravitational waveform model. Based off Ahmkel's [Keras-Project-Template](https://github.com/Ahmkel/Keras-Project-Template).

# Table of contents

- [Scripts](#scripts)
- [Configuration file structure](#config-file-structure)
- [Project architecture](#project-architecture)
- [Template Details](#template-details)
    - [Project Architecture](#project-architecture)
    - [Folder Structure](#folder-structure)
    - [Main Components](#main-components)
- [Future Work](#future-work)
- [Example Projects](#example-projects)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

<a name="scripts"></a>
# Scripts 

All the presented results can be reproduced from this project. The scripts in the root folder are:

- <h4>Training script

Script's name is `main.py`. It parses all the information in the config.json file then declares and trains the model. Usage:

```
python main.py -c [path to configuration file]
```

- <h4>Timing script

Script's name is `timing_script.py`. It loads trained model and uses python's `timeit` library to time the generation time for each model (and surrogate models). Usage:

```
python timing_script.py [-h] [-gm] [-pu] [-bs] [-d] [-n] [-en]

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

## configs

The configs directory holds the .json configuration files.

## experiment_logs

Directory for the training logs.

## experiments

Directory for the saved models. Every model is saved in a folder with the datetime and hour when the training starts. In this directory, another folder is created with the name specified in the configuration file. This folder contains two folders:

1. A checkpoint folder with the weights for every best model trained, (joblib's) serialized files with the training losses for each training performed and the particular configuration file used for the experiment.
2. A tensorboard directory for the usage of the Tensorboard callback.

## figures

Directory containing relevant figures and mismatch files for particular models. The directory name references the model used and the dataset in which it was trained.

- dense/mapper/reg reference a dense NN, the mapped autoencoder and the regularized autoencoder.
- q/qz/qzp reference BBH datasets in which the mass ratio is used (q), mass ratio + orbit-aligned spin (qz) and mass ratio + randomly aligned spin (qzp).
- q/q_lbd reference BNS datasets with mass ratio (q) or mass ratio + tidal deformabilities (q_lbd).

In each folder, a serialized (using joblib) binary file with train and test set mismatches is stored. Two sub-directories with train and test figures are also in this folder. Each contains the corresponding mismatch histogram and four waveform plots in which a comparison between the generated waveform and the ground truth are compared. Each of the four plots correspond to the worst and best case scenarios, the waveform corresponding to the mismatch at the 50th percentile of the mismatch distribution and the one corresponding to the 10th percentile.

## base

Directory containing abstract definitions for the data loader, trainer and model classes. Each of the defined classes for these purposes are inherited from the base directory.

## data_loader

Directory containing the data loader classes. It has code for data generation for the training loop, logic for the data loading and some preprocessing utilities, such as standard scaling,, shuffling and train-test splits.

## models

Directory containing the declaration of all the used neural networks and the overall architectures for each proposed model.

## trainers

Directory containing the classes where all the callbacks are defined and the fit methods are called according to the configuration file.

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

# Comet.ml Integration
This template also supports reporting to Comet.ml which allows you to see all your hyper-params, metrics, graphs, dependencies and more including real-time metric.

Add your API key [in the configuration file](configs/simple_mnist_config.json#L15):


For example:  `"comet_api_key": "your key here"`

Here's how it looks after you start training:
<div align="center">

<img align="center" width="800" src="https://comet-ml.nyc3.digitaloceanspaces.com/CometDemo.gif">

</div>

You can also link your Github repository to your comet.ml project for full version control.


# Template Details

## Project Architecture

<div align="center">

<img align="center" width="600" src="https://github.com/Ahmkel/Keras-Project-Template/blob/master/figures/ProjectArchitecture.jpg?raw=true">

</div>


## Folder Structure

```
├── main.py             - here's an example of main that is responsible for the whole pipeline.
│
│
├── base                - this folder contains the abstract classes of the project components
│   ├── base_data_loader.py   - this file contains the abstract class of the data loader.
│   ├── base_model.py   - this file contains the abstract class of the model.
│   └── base_train.py   - this file contains the abstract class of the trainer.
│
│
├── model               - this folder contains the models of your project.
│   └── simple_mnist_model.py
│
│
├── trainer             - this folder contains the trainers of your project.
│   └── simple_mnist_trainer.py
│
|
├── data_loader         - this folder contains the data loaders of your project.
│   └── simple_mnist_data_loader.py
│
│
├── configs             - this folder contains the experiment and model configs of your project.
│   └── simple_mnist_config.json
│
│
├── datasets            - this folder might contain the datasets of your project.
│
│
└── utils               - this folder contains any utils you need.
     ├── config.py      - util functions for parsing the config files.
     ├── dirs.py        - util functions for creating directories.
     └── utils.py       - util functions for parsing arguments.
```


## Main Components

### Models
You need to:
1. Create a model class that inherits from **BaseModel**.
2. Override the ***build_model*** function which defines your model.
3. Call ***build_model*** function from the constructor.


### Trainers
You need to:
1. Create a trainer class that inherits from **BaseTrainer**.
2. Override the ***train*** function which defines the training logic.

**Note:** To add functionalities after each training epoch such as saving checkpoints or logs for tensorboard using Keras callbacks:
1. Declare a callbacks array in your constructor.
2. Define an ***init_callbacks*** function to populate your callbacks array and call it in your constructor.
3. Pass the callbacks array to the ***fit*** function on the model object.

**Note:** You can use ***fit_generator*** instead of ***fit*** to support generating new batches of data instead of loading the whole dataset at one time.

### Data Loaders
You need to:
1. Create a data loader class that inherits from **BaseDataLoader**.
2. Override the ***get_train_data()*** and the ***get_test_data()*** functions to return your train and test dataset splits.

**Note:** You can also define a different logic where the data loader class has a function ***get_next_batch*** if you want the data reader to read batches from your dataset each time.

### Configs
You need to define a .json file that contains your experiment and model configurations such as the experiment name, the batch size, and the number of epochs.


### Main
Responsible for building the pipeline.
1. Parse the config file
2. Create an instance of your data loader class.
3. Create an instance of your model class.
4. Create an instance of your trainer class.
5. Train your model using ".Train()" function on the trainer object.

### From Config
We can now load models without having to explicitly create an instance of each class. Look at:
1. from_config.py: this can load any config file set up to point to the right modules/classes to import
2. Look at configs/simple_mnist_from_config.json to get an idea of how this works from the config. Run it with:
```shell
python from_config.py -c configs/simple_mnist_from_config.json
```
3. See conv_mnist_from_config.json (and the additional data_loader/model) to see how easy it is to run a different experiment with just a different config file:
```shell
python from_config.py -c configs/conv_mnist_from_config.json
```

# Example Projects
* [Toxic comments classification using Convolutional Neural Networks and Word Embedding](https://github.com/Ahmkel/Toxic-Comments-Competition-Kaggle)


# Future Work
Create a command line tool for Keras project scaffolding where the user defines a data loader, a model, a trainer and runs the tool to generate the whole project. (This is somewhat complete now by loading each of these from the config)


# Contributing
Any contributions are welcome including improving the template and example projects.

# Acknowledgements
This project template is based on [MrGemy95](https://github.com/MrGemy95)'s [Tensorflow Project Template](https://github.com/MrGemy95/Tensorflow-Project-Template).


Thanks for my colleagues [Mahmoud Khaled](https://github.com/MahmoudKhaledAli), [Ahmed Waleed](https://github.com/Rombux) and [Ahmed El-Gammal](https://github.com/AGammal) who worked on the initial project that spawned this template.
