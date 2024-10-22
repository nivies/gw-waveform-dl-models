import os 
from subprocess import run
from utils.config import process_config
import tempfile
import json
    
for reg_coef in [1e-2, 1e-4, 1e-6]:

    _, config_dict = process_config("/home/nino/GW/Keras-Project-Template/configs/gw_mapped_ae_config.json")
    
    config_dict['exp']['name'] = f"Mapped_ae_l1_reg_{reg_coef}"
    config_dict['model']['model_id'] = "3"
    config_dict['model']['ae_id'] = "l1_regularized"
    config_dict['model']['reg_weight'] = reg_coef

    with tempfile.TemporaryDirectory() as tmpdirname:

        path_config = os.path.join(tmpdirname, "config.json")

        with open(path_config, 'w') as fp:
            json.dump(config_dict, fp)

        run(f"python3 main.py -c {path_config}", shell = True)
        run(f"python3 main.py -c {path_config}", shell = True)

        
for reg_coef in [1e-2, 1e-4, 1e-6, 1e-8]:

    _, config_dict = process_config("/home/nino/GW/Keras-Project-Template/configs/gw_mapped_ae_config.json")
    
    config_dict['exp']['name'] = f"Mapped_ae_custom_reg_{reg_coef}"
    config_dict['model']['model_id'] = "3"
    config_dict['model']['ae_id'] = "custom_regularized"
    config_dict['model']['reg_weight'] = reg_coef

    with tempfile.TemporaryDirectory() as tmpdirname:

        path_config = os.path.join(tmpdirname, "config.json")

        with open(path_config, 'w') as fp:
            json.dump(config_dict, fp)

        run(f"python3 main.py -c {path_config}", shell = True)
        run(f"python3 main.py -c {path_config}", shell = True)

main_path = "/home/nino/GW/Keras-Project-Template/experiments"

for folder in os.listdir(main_path):
    
    if "2024" not in folder:
        continue

    path = os.path.join(main_path, folder)
    path = os.path.join(path, os.listdir(path)[0])
    in_path = os.path.join(path, "checkpoints")

    # Transfer learning call
    run(f"python3 transfer_learning.py -ld {in_path} -ls overlap -e 2000", shell = True)

    # Retrain call
    run(f"python3 retrain.py -ld {in_path} -ls overlap -e 2000", shell = True)

    # Autoencoder evaluation call
    run(f"python3 evaluate_autoencoders.py -l {in_path}", shell = True)