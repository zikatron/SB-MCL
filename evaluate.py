import os.path as path
from argparse import ArgumentParser
from datetime import datetime
from glob import glob
from itertools import product
import csv

import torch
import yaml
from torch.utils.data import DataLoader

from dataset import DATASET
from models import MODEL
from models.model import Output
from train import get_config, prepare_data
from utils import Timer

parser = ArgumentParser()
parser.add_argument('--config', '-c')
parser.add_argument('--log-dir', '-l')
parser.add_argument('--override', '-o', default='')
parser.add_argument('--tasks', '-t', default='')
parser.add_argument('--shots', '-s', default='')

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def compute_class_range_accuracies(logits, labels, num_classes):
    """
    Compute accuracy for each class range (bins of size num_classes/10).
    
    Args:
        logits: [batch, test_num, num_classes] - model predictions
        labels: [batch, test_num] - ground truth labels
        num_classes: total number of classes
    
    Returns:
        list of accuracies, one per bin (10 bins total)
    """
    # Calculate bin size (assumes num_classes divisible by 10)
    bin_size = num_classes // 10
    num_bins = 10
    
    # Flatten batch dimension
    batch_size = logits.shape[0]
    test_num = logits.shape[1]
    logits_flat = logits.reshape(-1, num_classes)  # [batch*test_num, num_classes]
    labels_flat = labels.reshape(-1)  # [batch*test_num]
    
    # Get predictions
    predictions = logits_flat.argmax(dim=-1)  # [batch*test_num]
    
    # Calculate accuracy for each bin
    bin_accuracies = []
    for bin_idx in range(num_bins):
        start_class = bin_idx * bin_size
        end_class = (bin_idx + 1) * bin_size
        
        # Find examples in this class range
        mask = (labels_flat >= start_class) & (labels_flat < end_class)
        
        if mask.sum() > 0:
            # Calculate accuracy for this bin
            correct = (predictions[mask] == labels_flat[mask]).float()
            accuracy = correct.mean().item()
        else:
            # No examples in this range
            accuracy = None
        
        bin_accuracies.append(accuracy)
    
    return bin_accuracies


def save_class_range_csv(accuracies_dict, num_classes, csv_path):
    """
    Save per-class-range accuracies to CSV.
    
    Args:
        accuracies_dict: {episode_idx: [acc_bin0, acc_bin1, ..., acc_bin9]}
        num_classes: total number of classes
        csv_path: where to save the CSV
    """
    bin_size = num_classes // 10
    num_episodes = len(accuracies_dict)
    
    # Create CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header row
        header = ['class_range'] + [f'episode_{i+1}' for i in range(num_episodes)]
        writer.writerow(header)
        
        # Data rows (one per bin)
        for bin_idx in range(10):
            start_class = bin_idx * bin_size
            end_class = (bin_idx + 1) * bin_size - 1
            row = [f'{start_class}-{end_class}']
            
            # Add accuracy for each episode
            for episode_idx in range(num_episodes):
                acc = accuracies_dict[episode_idx][bin_idx]
                # Format: 4 decimal places, or empty if None
                row.append(f'{acc:.4f}' if acc is not None else '')
            
            writer.writerow(row)
    
    print(f'✅ Class range accuracies saved to {csv_path}')


def main():
    args = parser.parse_args()
    config_path = args.config if args.config is not None else path.join(args.log_dir, 'config.yaml')
    config = get_config(config_path)

    # Override options
    for option in args.override.split('|'):
        if not option:
            continue
        address, value = option.split('=')
        keys = address.split('.')
        here = config
        for key in keys[:-1]:
            if key not in here:
                here[key] = {}
            here = here[key]
        if keys[-1] not in here:
            print(f'Warning: {address} is not defined in config file.')
        here[keys[-1]] = yaml.load(value, Loader=yaml.FullLoader)

    config['log_dir'] = args.log_dir

    # Evaluation settings
    eval_settings = []
    eval_settings.extend([(int(t), config['train_shots']) for t in args.tasks.split(',')])
    eval_settings.extend([(config['tasks'], int(s)) for s in args.shots.split(',')])
    print(f'Evaluation settings: {eval_settings}')

    evaluate(0, config, eval_settings)


def evaluate(rank, config, eval_settings):
    # Build model
    model = MODEL[config['model']](config).to(rank)

    # Resume checkpoint
    ckpt_paths = sorted(glob(path.join(config['log_dir'], 'ckpt-*.pt')))
    if len(ckpt_paths) == 0:
        raise RuntimeError(f'No checkpoint found in {config["log_dir"]}')
    ckpt_path = ckpt_paths[-1]
    # Get step number from checkpoint name
    ckpt_step = int(path.basename(ckpt_path).split('-')[1].split('.')[0])
    if ckpt_step != config['max_train_steps']:
        raise RuntimeError(f'Latest checkpoint {ckpt_path} does not match max_train_steps {config["max_train_steps"]}')
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model'], strict=False)
    print(f'Checkpoint loaded from {ckpt_path}')
    model.eval()

    # Data
    Dataset = DATASET[config['dataset']]
    meta_splits = ['test']
    datasets = {
        split: Dataset(config, root='./data', meta_split=split)
        for split in meta_splits
    }

    start_time = datetime.now()
    print(f'Computation started at {start_time}')

    for (tasks, shots), meta_split in product(eval_settings, meta_splits):
        print(f'Evaluation with {tasks}-task {shots}-shot meta-{meta_split} set')
        result_path = path.join(config['log_dir'], f'evaluation-{tasks}t{shots}s-meta_{meta_split}.pt')
        
        # NEW: CSV path for class range accuracies
        csv_path = path.join(config['log_dir'], f'class_range_accuracy-{tasks}t{shots}s-meta_{meta_split}.csv')
        
        if path.exists(result_path):
            print(f'Already evaluated in {result_path}')
            continue
        if config['dataset'] in ['omniglot', 'celeb']:
            if tasks > 500:
                print(f'Skip {tasks}t{shots}s because it exceeds 500 tasks')
                continue
            if shots > 10:
                print(f'Skip {tasks}t{shots}s because it exceeds 10 shots')
                continue
        config['tasks'] = tasks
        config['train_shots'] = shots
        config['test_shots'] = min(max((1000 // tasks), 1), 10 if 'omniglot' in config['dataset'] or 'celeb' in config['dataset'] else 50)
        config['train_chunk'] = 1000
        total_episodes = 512
        max_train_len = 10_000
        if config['tasks'] * config['train_shots'] > max_train_len:
            print(f'Skip {tasks}t{shots}s because it exceeds max_ex_per_epi={max_train_len}')
            continue
        ex_per_epi = config['tasks'] * (config['train_shots'] + config['test_shots'])
        max_ex_per_batch = 15_000 if 'max_ex_per_batch' not in config else config['max_ex_per_batch']
        eval_batch_size = min(max_ex_per_batch // ex_per_epi, total_episodes)
        if eval_batch_size == 0:
            raise RuntimeError(f'Too large episode size: {tasks}t{shots}s')
        eval_batch_size = 2 ** int(eval_batch_size.bit_length() - 1)  # round down to power of 2
        eval_iter = total_episodes // eval_batch_size
        data_loader = DataLoader(
            datasets[meta_split], batch_size=eval_batch_size, collate_fn=datasets[meta_split].collate_fn)
        data_loader_iter = iter(data_loader)

        # NEW: Dictionary to store per-episode class range accuracies
        class_range_accuracies = {}
        episode_counter = 0

        with torch.no_grad(), Timer('Evaluation time: {:.3f}s'):
            output = Output()
            print('-' * eval_iter + f' batch_size={eval_batch_size}')
            for _ in range(eval_iter):
                train_x, train_y, test_x, test_y = next(data_loader_iter)
                train_x, train_y, test_x, test_y = prepare_data(train_x, train_y, test_x, test_y, rank=rank)

                batch_output = model(train_x, train_y, test_x, test_y, summarize=True, meta_split='test')
                output.extend(batch_output)
                
                # NEW: Extract logits and labels from this batch to compute class range accuracies
                if 'logit' in batch_output:
                    batch_logits = batch_output['logit']  # [batch_size, test_num, num_classes]
                    batch_labels = test_y  # [batch_size, test_num]
                    
                    # Process each episode in the batch separately
                    for i in range(batch_logits.shape[0]):
                        episode_logits = batch_logits[i:i+1]  # [1, test_num, num_classes]
                        episode_labels = batch_labels[i:i+1]  # [1, test_num]
                        
                        # Compute class range accuracies for this episode
                        accuracies = compute_class_range_accuracies(
                            episode_logits, episode_labels, config['tasks']
                        )
                        
                        # Store with episode index
                        class_range_accuracies[episode_counter] = accuracies
                        episode_counter += 1
                
                print('.', end='', flush=True)
            
            # Save original evaluation results
            torch.save(output.export(), result_path)
            print()
            
            # NEW: Save class range accuracies to CSV
            if class_range_accuracies:
                save_class_range_csv(class_range_accuracies, config['tasks'], csv_path)

    end_time = datetime.now()
    print()
    print(f'Evaluation ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')
    with open(path.join(config['log_dir'], 'eval_completed.yaml'), 'a') as f:
        yaml.dump({
            'end_time': end_time,
        }, f)


if __name__ == '__main__':
    main()