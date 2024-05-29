import tensorflow as tf
from utils.args import get_args_eval
from utils.plot_utils import make_plots
from utils.eval import load_model_and_data_loader
from utils.config import get_config_from_json
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print("\n")

def main():

    try:
        args = get_args_eval()
    except:
        print("Missing or invalid arguments")
        exit(0)

    model, data_loader = load_model_and_data_loader(args.load_dir) 
    config, _ = get_config_from_json(os.path.join(args.load_dir, "config.json"))       

    print("Making plots")
    make_plots(model = model, dir = args.opt_dir, data_loader = data_loader, config = config, metric = args.metric)


if __name__ == '__main__':
    main()
