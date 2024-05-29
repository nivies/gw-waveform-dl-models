import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args

def get_args_eval():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-d', '--output_directory',
        dest='opt_dir',
        metavar='D',
        default='None',
        help='Output directory for the plots'
    )    
    argparser.add_argument(
        '-l', '--load_checkpoint',
        dest='load_dir',
        metavar='LC',
        default='None',
        help='Directory of checkpoint to load model'
    )
    argparser.add_argument(
        '-m', '--metric',
        dest='metric',
        default="real", 
        metavar="M",
        help="Whether to use the real/imaginary part of the waveform or the full complex waveform. Must be \"real\", \"imag\" or \"overlap\"."
    )
    args = argparser.parse_args()
    return args