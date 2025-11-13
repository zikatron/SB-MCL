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
args = parser.parse_args()

