import os
import json
from utils.config import process_config
from subprocess import run
import tempfile

l_dim = 5
model_ids_map = [1, 0, 0]
model_ids_ae = [0, 0, 1]

for id_map, id_ae in zip(model_ids_map, model_ids_ae):

    config, config_dict = process_config("/home/nino/GW/Keras-Project-Template/configs/gw_mapped_ae_config.json")

    config_dict['model']['model_id'] = str(id_map)
    config_dict['model']['ae_id'] = str(id_ae)
    config_dict['model']['latent_dim'] = l_dim
    config_dict['trainer']['uninitialised'] = True
    config_dict['exp']['name'] = f"qz_ae_ldim{l_dim}_mapid{id_map}_aeid{id_ae}_uninit"


    with tempfile.TemporaryDirectory() as tmpdirname:

        path = os.path.join(tmpdirname, "config_pca.json")

        with open(path, 'w') as fp:
            json.dump(config_dict, fp)

        run(f"python3 main.py -c {path}", shell = True)
        run(f"python3 main.py -c {path}", shell = True)

l_dim = 5
model_ids_map = [2, 3]
model_ids_ae = [0, 1]

for id_map in model_ids_map:
    for id_ae in model_ids_ae:

        config, config_dict = process_config("/home/nino/GW/Keras-Project-Template/configs/gw_mapped_ae_config.json")

        config_dict['model']['model_id'] = str(id_map)
        config_dict['model']['ae_id'] = str(id_ae)
        config_dict['model']['latent_dim'] = l_dim
        config_dict['trainer']['uninitialised'] = False
        config_dict['exp']['name'] = f"qz_ae_ldim{l_dim}_mapid{id_map}_aeid{id_ae}"


        with tempfile.TemporaryDirectory() as tmpdirname:

            path = os.path.join(tmpdirname, "config_pca.json")

            with open(path, 'w') as fp:
                json.dump(config_dict, fp)

            run(f"python3 main.py -c {path}", shell = True)
            run(f"python3 main.py -c {path}", shell = True)

l_dim = 15
model_ids_map = [0, 1, 2, 3]
model_ids_ae = [0, 1]

for id_map in model_ids_map:
    for id_ae in model_ids_ae:

        config, config_dict = process_config("/home/nino/GW/Keras-Project-Template/configs/gw_mapped_ae_config.json")

        config_dict['model']['model_id'] = str(id_map)
        config_dict['model']['ae_id'] = str(id_ae)
        config_dict['model']['latent_dim'] = l_dim
        config_dict['trainer']['uninitialised'] = False
        config_dict['exp']['name'] = f"qz_ae_ldim{l_dim}_mapid{id_map}_aeid{id_ae}"


        with tempfile.TemporaryDirectory() as tmpdirname:

            path = os.path.join(tmpdirname, "config_pca.json")

            with open(path, 'w') as fp:
                json.dump(config_dict, fp)

            run(f"python3 main.py -c {path}", shell = True)
            run(f"python3 main.py -c {path}", shell = True)

