from datetime import datetime
import os

import numpy as np
import torch

BASE_PATH = './results'

def init_exp_dir(args):
    dir_path = BASE_PATH
    dir_path += '/debug' if args.debug else ''
    dir_path += f"/{args.dataset}/buffer_{args.buffer}/{args.algorithm.replace(' ', '_')}"

    if args.sel_strategy is not None:
        dir_path += f"/{args.sel_strategy}"
        if args.sel_strategy == 'teal' and args.teal_type is not None:
            dir_path += f"_{args.teal_type}"
    if args.inc_model != 'resnet18':
        dir_path += f"/{args.inc_model}"
    if args.seed is not None:
        dir_path += f"/seed_{args.seed}"

    if not os.path.isdir(dir_path):
        try:
            os.makedirs(dir_path)
        except FileExistsError:
            pass

    return dir_path


def init_exp_folder(dir_path, exp_name=None):
    if exp_name is None:
        now = datetime.now()
        exp_name = f'{now.year}_{now.month}_{now.day}_' \
                   f'{str(now.hour).zfill(2)}{str(now.minute).zfill(2)}{str(now.second).zfill(2)}'
    counter = 0
    while True:
        exp_path = f'{dir_path}/{exp_name}_{chr(97 + counter)}'
        try:
            os.mkdir(exp_path)
            break
        except FileExistsError:
            counter += 1
            pass
        except FileNotFoundError:
            os.mkdir(dir_path)
            counter += 1

    print(f"Experiment Directory is {exp_path}.\n")
    return exp_path
