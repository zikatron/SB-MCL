import csv
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
    config['codebook_size'] = 128
    config['num_codebooks'] = 32
    config['num_channels'] = 512
    config['lr'] = 1e-2
    config['optim_args'] = {'lr': config['lr'], 'weight_decay': 1e-5, 'momentum': 0.9}
    config['replay_buffer_size'] = 7_500
    config['batch_size'] = 64
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
    model.load_state_dict(state_dict, strict=True)
    model = model.cuda()

    # Prepare data
    Dataset = DATASET[config['dataset']]
    meta_test_set = Dataset(config, root='./data', meta_split='test',seed_classes=args.run_number, model=config['model'])
    meta_test_loader = DataLoader(
        meta_test_set,
        batch_size=1,
        num_workers=0,
        collate_fn=meta_test_set.collate_fn)
    
    init_data, cl_data = next(iter(meta_test_loader))

    train_x_initialise, train_y_initialise, test_x_initialise, test_y_initialise = init_data
    train_x_cl, train_y_cl, test_x_cl, test_y_cl = cl_data

    num_init_classes = torch.max(train_y_initialise) + 1
    print(f"Detected {num_init_classes} initial classes. Offsetting CL labels by this amount...")

    # PyTorch broadcasts the scalar (500) to every element in the [1, N] tensors
    train_y_cl += num_init_classes
    test_y_cl += num_init_classes

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
    model.mlp.train()
    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = start + chunk_size
        
        # Extract chunk - need [0] to get [8400, ...] then slice
        chunk_x = train_x_cl[0, start:end] 
        chunk_y = train_y_cl[0, start:end]  # Shape: [21]

        outputs_train = model(chunk_x, chunk_y, summarize=True, split='train')
        # Aggregate losses and accuracies from the list of outputs
        losses = torch.stack([out['loss/train'] for out in outputs_train])
        accs = torch.stack([out['top1_acc/train'] for out in outputs_train])
        
        # Log to wandb
        # wandb_logger.log({
        #     'train_loss': losses.mean().item(),
        #     'train_acc': accs.mean().item(),
        #     'chunk': chunk_idx
        # }, step=chunk_idx)
        
        # Print progress
        # print(f"Chunk {chunk_idx+1}/{num_chunks} | Loss: {losses.mean():.4f} | Acc: {accs.mean():.4f}")
        
        # Clean up
        del outputs_train, losses, accs
        torch.cuda.empty_cache()
    # Close wandb
    wandb.finish()
    print("\n--- Starting Final Evaluation ---")
    model.mlp.eval()
    model.decoder.eval()
    
    # --- [START] 1. Get Raw Predictions ---
    with torch.no_grad():
        # Call with summarize=False to get the single Output object
        # with our 'predictions' and 'labels' fields
        output_data = model(test_x_combined[0], test_y_combined[0], summarize=False, split='test')

    # Extract the raw data (move to CPU for numpy/list operations)
    all_preds = output_data['predictions'].cpu()   # Shape: [N]
    all_labels = output_data['labels'].cpu()      # Shape: [N]

    # === [START] SANITY CHECK ===
    print("\n--- Sanity Check: Label Inspection ---")
    
    # 1. Check Unique Labels
    unique_labels = torch.unique(all_labels)
    print(f"Found {len(unique_labels)} unique labels.")
    print(f"Min label: {unique_labels.min().item()}, Max label: {unique_labels.max().item()}")
    
    # 2. Check Order
    # This will show you the pattern. 
    # E.g., if you have 10 shots, you expect 10 zeros, then 10 ones, etc.
    print("\nFirst 50 labels in order:")
    print(all_labels[:50])
    
    print("\nLast 50 labels in order:")
    print(all_labels[-50:])
    
    # 3. Check where the init/CL data split happens
    # We concatenated test_x_initialise and test_x_cl
    # Find the size of the initial test set
    init_test_size = test_x_initialise.shape[1] 
    print(f"\nInitial test set size: {init_test_size} samples")
    print("Labels at the end of the initial data:")
    print(all_labels[init_test_size - 25 : init_test_size + 25])
    print("--- End of Sanity Check ---")
    # === [END] SANITY CHECK ===
    
    # Get the single, overall test loss
    test_loss = output_data['loss/test'].mean().item()
    # --- [END] 1. Get Raw Predictions ---

    
    # --- [START] 2. Manually Calculate Metrics ---
    
    # Calculate overall accuracy
    correct = (all_preds == all_labels)
    overall_accuracy = correct.float().mean().item()
    
    print(f"\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")

    # Calculate interval accuracies
    interval_accuracies = []
    num_classes_per_interval = 100
    
    # Assuming 1000 classes total, for 10 intervals
    num_intervals = (all_labels.max().item() + 1) // num_classes_per_interval
    
    print(f"Calculating {num_intervals} interval accuracies...")

    for i in range(num_intervals):
        start_class = i * num_classes_per_interval
        end_class = (i + 1) * num_classes_per_interval # e.g., 0-100, 100-200

        # Find all samples where the TRUE label is in this interval
        mask = (all_labels >= start_class) & (all_labels < end_class)
        
        if mask.sum() == 0:
            # Avoid division by zero if an interval has no test samples
            interval_acc = 0.0 
        else:
            # Get the 'correct' results for JUST this interval's samples
            interval_correct = correct[mask]
            interval_acc = interval_correct.float().mean().item()
        
        interval_accuracies.append(interval_acc)
        print(f"  Interval {start_class}-{end_class-1} Acc: {interval_acc:.4f}")
    # --- [END] 2. Manually Calculate Metrics ---


    # --- [START] 3. Write to CSV ---
    
    # Define a new CSV file name for these detailed results
    parent_dir = path.dirname(args.log_dir)
    csv_path = path.join(parent_dir, 'combined_cl_results_detailed.csv')
    
    file_exists = path.isfile(csv_path)
    
    try:
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # 1. Write the dynamic header
            if not file_exists:
                headers = ['run_name', 'seed', 'test_loss', 'overall_accuracy']
                # Add a header for each interval
                for i in range(num_intervals):
                    start = i * num_classes_per_interval
                    end = start + num_classes_per_interval - 1
                    headers.append(f"interval_{start}-{end}")
                writer.writerow(headers)
                
            # 2. Write the dynamic data row
            data_row = [args.run_name, args.run_number, f"{test_loss:.4f}", f"{overall_accuracy:.4f}"]
            # Add the result for each interval
            for acc in interval_accuracies:
                data_row.append(f"{acc:.4f}")
            writer.writerow(data_row)
            
        print(f"Successfully saved detailed results to {csv_path}")
        
    except Exception as e:
        print(f"Error saving results to CSV: {e}")


def main():
    config = load_config(args.config)
    train_remind(config)

if __name__ == '__main__':
    main()