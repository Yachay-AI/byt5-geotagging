from argparse import ArgumentParser
from aux_files.learner import Learner


parser = ArgumentParser()
# script
parser.add_argument('--data_dir', default='./data', action='store')
parser.add_argument('--arch', default='char_lstm', choices=['char_lstm', 'char_cnn'])
parser.add_argument('--run_name', default=None)
parser.add_argument('--save_prefix', default='model', help='Path to save the model to')
parser.add_argument('--subsample', action='store', type=int, default=9e9)
# data
parser.add_argument('--split_uids', action='store_true', help='Hold-out train user IDs when generating validation')
parser.add_argument('--max_seq_len', default=-1, type=int, help='Truncate tweet text to length; -1 to disable')
parser.add_argument('--truncate_ratio', default=-1, help='Subsample dataset to this ratio')
parser.add_argument('--data_format', action='store', choices=['default', 'old', 'new'], default='new')
# training
parser.add_argument('--loss', default='mse', choices=['mse', 'l1', 'smooth_l1'])
parser.add_argument('--optimizer', default='adam', choices=['adam', 'adamw', 'sgd'])
parser.add_argument('--metric', default='gc_distance')
parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                    help='Number of training steps to accumulate gradient over')
parser.add_argument('--scheduler', default='constant', choices=['linear', 'constant'],
                    help="`linear' scales up the LR over `args.warmup_ratio' steps")
parser.add_argument('--warmup_ratio', default=0.2, type=float,
                    help="Ratio of maximum training steps to scale LR over")
parser.add_argument('--lr', type=float, default=1e-4, help="Optimiser learning rate")
parser.add_argument('--dropout', type=float, default=0.3, help="Model dropout ratio")
parser.add_argument('--batch_size', type=int, default=128, help="Training batch size")
parser.add_argument('--num_epochs', type=int, default=10, help="Number of training epochs")
# model
parser.add_argument('--reduce_layer', action='store_true', default=False,
                    help="Add linear layer before output")
parser.add_argument('--conf_estim', action='store_true', default=False,
                    help="Use confidence estimation")
parser.add_argument('--conf_p', action='store', default=0.6, type=float,
                    help="Estimation hyperparam; top-p% = 1")
parser.add_argument('--conf_lambda', action='store', default=0.1, type=float,
                    help="Confidence loss weight")
parser.add_argument('--confidence_bands', nargs='+', type=int, default=[2, 5, 10, 25, 50, 100],
                    help="Output scores for specified confidence bands")
parser.add_argument('--model_save_band', type=int, default=25,
                    help="Save model based on performance in band")
parser.add_argument('--no_conf_epochs', action='store', default=0.2, type=float,
                    help="Proportion of total no. of epochs after which to start co-training confidence estimator")
parser.add_argument('--confidence_validation_criterion', action='store_true', default=False,
                    help="Save models based on the score on the highest confidence level instead of on all data")
args = parser.parse_args()

conf_geo_learner = Learner(vars(args))

conf_geo_learner.train()