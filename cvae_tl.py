import os
from subprocess import run

path = "/home/nino/GW/Keras-Project-Template/experiments/cVAEs"

for model in os.listdir(path):

    model_path = os.path.join(path, model, "checkpoints")

    run(f"python3 transfer_learning.py -ld {model_path} -ls overlap -e 3000", shell = True)