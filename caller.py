import os
import json
from utils.config import process_config
from subprocess import run
import tempfile

pca_components = [5, 15]
model_ids = list(range(4))

with open("./caller_log.txt", 'w') as f:
    f.write(f"Training for N PCA components: {pca_components}\nTraining for models: {model_ids}\n------------------------------------------------\n\n")


for comp in pca_components:
    for id in model_ids:

        with open("./caller_log.txt", 'a') as f:
            f.write(f"N PCA components: {comp}\nModel ID     : {id}\n\n")

        config_pca_uninit, config_dict_pca_uninit = process_config("/home/nino/GW/Keras-Project-Template/configs_copy/gw_config_pca_0_uninit.json")
        config_pca, config_dict_pca = process_config("/home/nino/GW/Keras-Project-Template/configs_copy/gw_config_pca_0.json")

        config_dict_pca['model']['model_id'] = id
        config_dict_pca_uninit['model']['model_id'] = id

        config_dict_pca['model']['pca_n_components'] = comp
        config_dict_pca_uninit['model']['pca_n_components'] = comp

        config_dict_pca['exp']['name'] = f"qz_pca_n{comp}_id{id}"
        config_dict_pca_uninit['exp']['name'] = f"qz_pca_uninit_n{comp}_id{id}"

        with tempfile.TemporaryDirectory() as tmpdirname:

            path_pca = os.path.join(tmpdirname, "config_pca.json")
            path_pca_uninit = os.path.join(tmpdirname, "config_pca_uninit.json")

            with open(path_pca, 'w') as fp:
                json.dump(config_dict_pca, fp)

            with open(path_pca_uninit, 'w') as fp:
                json.dump(config_dict_pca_uninit, fp)

            run(f"python3 main.py -c {path_pca}", shell = True)
            run(f"python3 main.py -c {path_pca}", shell = True)
            run(f"python3 main.py -c {path_pca_uninit}", shell = True)
            run(f"python3 main.py -c {path_pca_uninit}", shell = True)
            

        

