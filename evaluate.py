import tensorflow as tf
import argparse
from utils.plot_utils import make_plots, make_plots_sxs
from utils.eval import load_model_and_data_loader
from utils.config import get_config_from_json
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print("\n")

def main():

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-d', '--output_directory', dest='opt_dir', metavar='D', default='None', help='Output directory for the plots')
    argparser.add_argument('-l', '--load_checkpoint', dest='load_dir', metavar='LC', default='None', help='Directory of checkpoint to load model')
    argparser.add_argument('--sxs', dest = 'sxs', metavar = "TL", default = False, help = '--sxs if the data to evaluate is from the SXS dataset, --no-sxs otherwise.', action=argparse.BooleanOptionalAction)
    argparser.add_argument('-m', '--metric', dest='metric', default="real", metavar="M", help="Whether to use the real/imaginary part of the waveform or the full complex waveform. Must be \"real\", \"imag\" or \"overlap\".")
    args = argparser.parse_args()

    if args.sxs:

        model, data_loader = load_model_and_data_loader(args.load_dir) 
        config, _ = get_config_from_json(os.path.join(os.path.dirname(args.load_dir), "config.json"))       

        print("Making plots")
        make_plots_sxs(model = model, dir = args.opt_dir, config = config, metric = args.metric)
    else:

        model, data_loader = load_model_and_data_loader(args.load_dir) 
        config, _ = get_config_from_json(os.path.join(args.load_dir, "config.json"))

        print("Making plots")
        make_plots(model = model, dir = args.opt_dir, data_loader = data_loader, config = config, metric = args.metric)


if __name__ == '__main__':
    main()
