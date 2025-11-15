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

from dataset import MetaCasia
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
parser.add_argument('--run-name', '-rn', default='remind_cl_stage')
parser.add_argument("--run_number", "-s", default=1)
args = parser.parse_args()

os.environ['WANDB_API_KEY'] = 'af68f61230db91e3ba854d69c29437700c715fc4'


def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    # Ensure these values are added to the configuration
    config['codebook_size'] = 32
    config['num_codebooks'] = 64
    config['num_channels'] = 512
    config['lr'] = 1e-2
    config['replay_buffer_size'] = 7500
    config['batch_size'] = 16
    return config

def train_remind(config):
    # Initialize model
    # Load pre-trained weights (with DDP prefix removal)
    ckpt = torch.load(path.join(args.log_dir, 'ckpt-best_model.pt'))
    state_dict = ckpt['model']
    if any(k.startswith('module.') for k in state_dict.keys()):
        print("Detected DDP checkpoint, removing 'module.' prefix...")
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    # Build REMIND model
    model = REMIND(config)
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda()

    # Prepare data
    MetaCasia.x_dict = None  # Reset class variable
    MetaCasia.meta_train_classes = None
    MetaCasia.meta_test_classes = None
    Dataset = DATASET[config['dataset']]
    meta_test_set = Dataset(config, root='./data', meta_split='test',seed_classes=args.run_number, model=config['model'])
    meta_test_loader = DataLoader(
        meta_test_set,
        batch_size=config['eval_batch_size'],
        num_workers=0,
        collate_fn=meta_test_set.collate_fn)
    
    init_data, cl_data = next(iter(meta_test_loader))

    train_x_initialise, train_y_initialise, test_x_initialise, test_y_initialise = init_data
    train_x_cl, train_y_cl, test_x_cl, test_y_cl = cl_data

    #moving onto CUDA
    train_x_initialise = train_x_initialise.cuda()
    train_y_initialise = train_y_initialise.cuda()
    test_x_initialise = test_x_initialise.cuda()
    test_y_initialise = test_y_initialise.cuda()
    train_x_cl = train_x_cl.cuda()
    train_y_cl = train_y_cl.cuda()
    test_x_cl = test_x_cl.cuda()
    test_y_cl = test_y_cl.cuda()

    train_x_initialise, train_y_initialise,test_x_initialise, test_y_initialise, train_x_cl, train_y_cl, test_x_cl, test_y_cl = prepare_data(train_x_initialise, train_y_initialise, test_x_initialise, test_y_initialise, train_x_cl, train_y_cl, test_x_cl, test_y_cl, rank=0)
    # Concatenate along dim=1 (the sample dimension)
    test_x_combined = torch.cat([test_x_initialise, test_x_cl], dim=1) 
    test_y_combined = torch.cat([test_y_initialise, test_y_cl], dim=1)

    # wandb initialization
    # wandb_logger = init_wandb(
    #     config, 
    #     name=f"REMIND_CL_{args.run_name}_seed{args.run_number}"
    # )

    # Intialise Model's PQ and Replay Buffer
    model.initialise_buffer_with_initial_data(train_x_initialise[0], train_y_initialise[0])
    
    # Pass ALL 840 classes at once (8400 samples if 10-shot)
    chunk_size = 10
    num_chunks = train_x_cl.shape[1] // chunk_size  # Shape is [1, 8400, ...]

    print("the num of chunks is", num_chunks)
    
    # for chunk_idx in range(num_chunks):
    #     start = chunk_idx * chunk_size
    #     end = start + chunk_size
        
    #     # Extract chunk - need [0] to get [8400, ...] then slice
    #     chunk_x = train_x_cl[0, start:end] 
    #     chunk_y = train_y_cl[0, start:end]  # Shape: [21]

    #     outputs_train = model(chunk_x, chunk_y, summarize=True, split='train')
    #     # Aggregate losses and accuracies from the list of outputs
    #     losses = torch.stack([out['loss/train'] for out in outputs_train])
    #     accs = torch.stack([out['acc/train'] for out in outputs_train])
        
    #     # Log to wandb
    #     wandb_logger.log({
    #         'train_loss': losses.mean().item(),
    #         'train_acc': accs.mean().item(),
    #         'chunk': chunk_idx
    #     }, step=chunk_idx)
        
    #     # Print progress
    #     # print(f"Chunk {chunk_idx+1}/{num_chunks} | Loss: {losses.mean():.4f} | Acc: {accs.mean():.4f}")
        
    #     # Clean up
    #     del outputs_train, losses, accs
    #     torch.cuda.empty_cache()
    # # Close wandb
    # wandb.finish()
    model.eval()
    with torch.no_grad():
        outputs_test = model(test_x_initialise[0], test_y_initialise[0], summarize=True, split='test')
    # Extract test metrics
    test_loss = outputs_test[0]['loss/test'].mean().item()  # outputs_test is a list
    test_acc = outputs_test[0]['acc/test'].mean().item()

    print(f"\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    # TODO: need to write this to a file from all the different runs. Should be CSV file.

def main():
    config = load_config(args.config)
    train_remind(config)

if __name__ == '__main__':
    main()