import argparse
import json

import torch
from cl_pipleine import ContinualLearningPipeline
from utils.utils import init_exp_dir, init_exp_folder


def add_default_args(parser):
    parser.add_argument('--cuda_id', dest='cuda_id', required=False, type=int, default=0)
    parser.add_argument('--inc_model', dest='inc_model', help='The model to train with (etc. resnet18)',
                        choices=['resnet18', 'arch_craft', 'slim_resnet18'],
                        required=False, type=str, default='resnet18')
    parser.add_argument('--num_epochs', dest='num_epochs', help='Number of epochs', required=False, type=int,
                        default=100)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size', required=False, type=int,
                        default=128)
    parser.add_argument('--momentum', dest='momentum', required=False, type=float, default=0.9)
    parser.add_argument('--weight_decay', dest='weight_decay', required=False, type=float, default=0.0002)
    parser.add_argument('--lr', dest='lr', required=False, type=float, default=0.1)
    parser.add_argument('--seed', dest='seed', required=False, type=int, default=None,
                        help='Random seed for classes order. If None- class order is 0,1,2,...')
    return parser


def get_parser():
    parser = argparse.ArgumentParser(description='Class Incremental Learning')
    parser.add_argument('--dataset', dest='dataset', help='Dataset name', required=True, type=str,
                        choices=['cifar10', 'cifar100', 'tinyimg', 'cub200'])
    parser.add_argument('--num_experiences', dest='num_experiences', help='Number of experiences/tasks',
                        required=True, type=int)
    parser.add_argument('--algorithm', dest='algorithm', help='Name of the algorithm to use',
                        choices=['er_ace', 'er', 'finetune', 'naive_replay'],
                        required=True, type=str)
    parser.add_argument('--sel_strategy', dest='sel_strategy', help='Selection strategy', required=False,
                        choices=['herding', 'teal', 'rm', 'centered', 'random'],
                        type=str, default=None)
    parser.add_argument('--teal_type', dest='teal_type',
                        help="If sel_strategy is 'teal', the type of TEAL to use",
                        choices=['one_time', 'log_iterative'],
                        required=False, type=str, default='log_iterative')
    parser.add_argument('--buffer', dest='buffer', help='buffer size', required=True, type=int)
    parser.add_argument('--debug', dest='debug', help='Debug mode', required=False, action='store_true',
                        default=False)

    parser = add_default_args(parser)
    return parser


if __name__ == '__main__':

    args = get_parser().parse_args()
    if args.algorithm == 'er_ace':
        args.lr = 0.01
        args.batch_size = 10
    if args.dataset == 'cub200':
        args.num_epochs = 30
        args.batch_size = 16
    if args.debug:
        print('Debug mode')
        args.num_epochs = 3
    print(vars(args))

    exp_dir = init_exp_dir(args)
    print(f"Experiment directory: {exp_dir}")

    device = torch.device("cuda", args.cuda_id) if torch.cuda.is_available() else "cpu"

    pipeline = ContinualLearningPipeline(args, device)
    exp_acc_dict = pipeline.train()

    exp_folder = init_exp_folder(exp_dir)
    with open(f'{exp_folder}/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    torch.save({'exp_acc_dict': exp_acc_dict, 'args': vars(args)}, f"{exp_folder}/results.pyth")


