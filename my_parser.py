import argparse
import yaml 

def parse_args_train():
    # The first arg parser parses out only the --config argument, this argument is used to
    # load a yaml file containing key-values that override the defaults for the main parser below
    config_parser = parser = argparse.ArgumentParser(
        description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    parser = argparse.ArgumentParser(
        description='aberration')

    # General
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4, metavar='T',
        help='Number of threads to be used (default: 4)')
    # Dataset 
    parser.add_argument('--dataset', '-d',  type=str, default='psf',
                        help='dataset') 
    parser.add_argument('--channel', type=list, nargs='+', default=[0,1,2],
                        help='channel to include')
    parser.add_argument('--bias_val', type=list, nargs='+', default=[-1,0,1],
                        help='bias val')
    parser.add_argument('--data-size', type=int, default=500000, metavar='TB',
        help='Training batch size (default: 6)')
    parser.add_argument('--bias_z', type=int, default=4, metavar='TB',
        help='Training batch size (default: 6)')
    parser.add_argument('--zernike', type=list, nargs='+', default=[3,5,6,7],
                        help='zernike')
    parser.add_argument('--infer', action='store_true', default=False,
                        help='infer')
    parser.add_argument('--data_path', type=str, default='./db/proj-InLightenUs/aberration/conv/dagm/Class10/',
                        help='path to dataset (single string)')
    parser.add_argument('--image_size', default=128, type=int)
    parser.add_argument('--npts', default=97, type=int)
    # Model parameters
    parser.add_argument('--train-batch', type=int, default=32, metavar='TB',
        help='Training batch size (default: 6)')
    parser.add_argument('--valid-batch', type=int, default=1, metavar='VB',
        help='Validation batch size (default: 1)')
    parser.add_argument('--train_size', type=float, default=1.0)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--z_range', type=float, default=1.0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
        help='Learning rate (default: 0.001)')
                   
    parser.add_argument('--epochs', type=int, default=1000, metavar='E',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--checkpoint-hist', type=int, default=1, metavar='CH',
                        help='number of checkpoints to keep (default: 10)')
    parser.add_argument('--resume_path', default='', type=str,
                        help='Path for resume model.' )
    parser.add_argument('--save_every', type=int, default=20)

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text