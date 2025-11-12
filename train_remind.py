import os
import os.path as path
import shutil
import socket
from argparse import ArgumentParser
from datetime import datetime
from glob import glob
from modulefinder import ModuleFinder

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
from dataset import DATASET
from models import MODEL
from models.model import Output
from train import get_config, prepare_data
from utils import Timer,init_wandb
from models.remind import REMIND

parser = ArgumentParser()
parser.add_argument('--config', '-c')
parser.add_argument('--model-config', '-mc')
parser.add_argument('--data-config', '-dc')
parser.add_argument('--log-dir', '-l')
parser.add_argument('--override', '-o', default='')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--no-backup', action='store_true')
parser.add_argument('--run-name', '-rn', default='i_didnt_set_a_name_lol')
parser.add_argument("--run_number", "-s", default=0)

def build_replay_buffer(config, ckpt_path, model):

    ckpt = torch.load(ckpt_path)
    state_dict = ckpt['model']

    # Check if checkpoint has DDP prefix (from multi-GPU training)
    if any(k.startswith('module.') for k in state_dict.keys()):
        print("Detected DDP checkpoint, removing 'module.' prefix...")
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    # Load with strict=True to catch any remaining issues
    model.load_state_dict(state_dict)
    print(f'Checkpoint loaded from {ckpt_path}')
    model.inialize_replay_buffer(train_x, train_y)

    replay_buffer = None
    return replay_buffer