import os
import json
from utils.config import process_config
from subprocess import run
import tempfile

latent_dimensions = [5, 15]
model_ids_map = list(range(4))
model_ids_ae = list(range(2))

with open("./caller_ae_log.txt", 'w') as f:
    f.write(f"Training for latent dimensions: {latent_dimensions}\nTraining for mapper models: {model_ids_map}\nTraining for autoencoder models: {model_ids_ae}\n------------------------------------------------\n\n")


for l_dim in latent_dimensions:
    for id_map in model_ids_map:
        for id_ae in model_ids_ae:

            with open("./caller_ae_log.txt", 'a') as f:
                f.write(f"Latent dimensions: {l_dim}\nMaping model ID     : {id_map}\nAutoencoder model ID        : {id_ae}\n\n")

            config, config_dict = process_config("/home/nino/GW/Keras-Project-Template/configs/gw_mapped_ae_config.json")

            config_dict['model']['model_id'] = str(id_map)
            config_dict['model']['ae_id'] = str(id_ae)
            config_dict['model']['latent_dim'] = l_dim
            config_dict['exp']['name'] = f"qz_ae_ldim{l_dim}_mapid{id_map}_aeid{id_ae}"
            config_dict['trainer']['uninitialised'] = False

            with tempfile.TemporaryDirectory() as tmpdirname:

                path = os.path.join(tmpdirname, "config_pca.json")

                with open(path, 'w') as fp:
                    json.dump(config_dict, fp)

                run(f"python3 main.py -c {path}", shell = True)
                run(f"python3 main.py -c {path}", shell = True)

                path = os.path.join(tmpdirname, "config_pca_uninit.json")

                config_dict['trainer']['uninitialised'] = True
                config_dict['exp']['name'] = f"qz_ae_ldim{l_dim}_mapid{id_map}_aeid{id_ae}_uninit"

                with open(path, 'w') as fp:
                    json.dump(config_dict, fp)

                run(f"python3 main.py -c {path}", shell = True)
                run(f"python3 main.py -c {path}", shell = True)
