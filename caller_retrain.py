import os
from subprocess import run

experiments_path = "/home/nino/GW/Keras-Project-Template/experiments"

for fold in os.listdir(experiments_path):
    path = os.path.join(experiments_path, fold)
    
    for exp in os.listdir(path):
        
        if exp == "figures":
            continue
        
        in_path = os.path.join(path, exp)
        print_path = os.path.join(in_path, os.listdir(in_path)[0])

        if "uninit" in print_path:
            continue

        in_path = os.path.join(print_path, "checkpoints")

        print("\n\n" + print_path + "\n\n")

        run(f"python3 retrain.py -ld {in_path} -ls overlap -e 2000", shell = True)