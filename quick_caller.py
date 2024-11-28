import os 
from subprocess import run
from utils.config import process_config
import tempfile
import json

_, config_dict = process_config("/home/nino/GW/Keras-Project-Template/configs/gw_mapped_ae_config.json")

# # Deep mapped autoencoder call
# config_dict['exp']['name'] = "deep_residual_mapped_autoencoder"

# with tempfile.TemporaryDirectory() as tmpdirname:

#     path_config = os.path.join(tmpdirname, "config.json")

#     with open(path_config, 'w') as fp:
#         json.dump(config_dict, fp)

#     run(f"python3 main.py -c {path_config}", shell = True)

# Deep regularized mapped autoencoder call
config_dict['exp']['name'] = "deep_residual_regularized_mapped_autoencoder"
config_dict['model']['deep']['regularization'] = "custom"

with tempfile.TemporaryDirectory() as tmpdirname:

    path_config = os.path.join(tmpdirname, "config.json")

    with open(path_config, 'w') as fp:
        json.dump(config_dict, fp)

    run(f"python3 main.py -c {path_config}", shell = True)

# # Deep uninitialised mapped autoencoder call
# config_dict['exp']['name'] = "deep_residual_uninitialised_mapped_autoencoder"
# config_dict['model']['deep']['regularization'] = ""
# config_dict['trainer']['uninitialised'] = True

# with tempfile.TemporaryDirectory() as tmpdirname:

#     path_config = os.path.join(tmpdirname, "config.json")

#     with open(path_config, 'w') as fp:
#         json.dump(config_dict, fp)

#     run(f"python3 main.py -c {path_config}", shell = True)

# # Deep cVAE training call
# _, config_dict = process_config("/home/nino/GW/Keras-Project-Template/configs/gw_cvae_config.json")
# config_dict['exp']['name'] = "symmetric_deep_cVAE"

# with tempfile.TemporaryDirectory() as tmpdirname:

#     path_config = os.path.join(tmpdirname, "config.json")

#     with open(path_config, 'w') as fp:
#         json.dump(config_dict, fp)

#     run(f"python3 main.py -c {path_config}", shell = True)

# Retraining, transfer learning and autoencoder evaluation
main_path = "/home/nino/GW/Keras-Project-Template/experiments"

for folder in os.listdir(main_path):
    
    if "2024" not in folder:
        continue

    path = os.path.join(main_path, folder)
    path = os.path.join(path, os.listdir(path)[0])
    in_path = os.path.join(path, "checkpoints")

    # Autoencoder evaluation call
    if not os.path.isfile(os.path.join(path, "ae_figures/mismatches_test.bin")):
        
        run(f"python3 evaluate_autoencoder.py -l {in_path}", shell = True)

    # Retrain call
    if not os.path.isfile(os.path.join(path, "retraining_overlap/history.bin")):

        run(f"python3 retrain.py -ld {in_path} -ls overlap -e 8000", shell = True)

    # Transfer learning call
    if not os.path.isfile(os.path.join(path, "transfer_learning_overlap/history.bin")):
        
        tl_path = os.path.join(path, "retraining_overlap")
        run(f"python3 transfer_learning.py -ld {tl_path} -ls overlap -e 8000", shell = True)