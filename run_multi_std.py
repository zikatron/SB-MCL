
"""
Script to run multiple training runs with train_std.py and collect results.

Usage:
    # Offline mode (default)
    uv run run_multiple_std.py -c cfg/casia-std.yaml -n 20
    
    # Online mode
    uv run run_multiple_std.py -c cfg/casia-std.yaml -n 20 --online
"""

import subprocess
import yaml
import csv
from pathlib import Path
import argparse



def main():
    parser = argparse.ArgumentParser(description='Run multiple standard training experiments')
    parser.add_argument('--model', '-m',required=True, help='Model name')
    parser.add_argument('--config', '-c', required=True, help='Path to config file')
    parser.add_argument('--num-runs', '-n', type=int, default=2, help='Number of runs')
    parser.add_argument('--online', action='store_true', help='Use online learning mode')
    parser.add_argument('--run-name', '-rn', default='i_didnt_set_a_name_lol', help='Name of the run')
    parser.add_argument('--script-path', '-sp',default='train_std.py', help='Path to the training script')
    args = parser.parse_args()
    
    # Determine base directory based on mode
    mode = 'online' if args.online else 'offline'
    base_dir = Path('experiments') / f'{args.model}-{mode}'
    base_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = base_dir / 'results.csv'
    
    # Create CSV with header if it doesn't exist
    csv_exists = csv_path.exists()
    csv_file = open(csv_path, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    
    if not csv_exists:
        csv_writer.writerow([
            'run', 'best_step', 'elapsed_time', 'best_test_acc', 'best_test_loss',
            'final_test_acc', 'final_test_loss', 'log_dir'
        ])
        csv_file.flush()
    
    results = []
    
    for run in range(args.num_runs):
        log_dir = base_dir / f"seed_{run + 1}"
        
        print(f"Run {run + 1}/{args.num_runs}")
        
        # Build command
        cmd = [
            "uv", "run", args.script_path,
            "-c", args.config,
            "-l", str(log_dir),
            "-rn", args.run_name,
            "-s", str(run + 1),  # run number
            "--no-backup",
        ]
        
        # Add online mode override if needed
        if args.online:
            cmd.extend(["-o", "online=True"])
        
        # Run training
        subprocess.run(cmd, check=True)

        
        # Read results from completed.yaml
        completed_path = log_dir / "completed.yaml"
        with open(completed_path, 'r') as f:
            result = yaml.safe_load(f)
        
        # Extract data
        run_data = {
            'run': run,
            'best_step': result['best_step'],
            'elapsed_time': result['elapsed_time'],
            'best_test_acc': result['best_test_acc'],
            'best_test_loss': result['best_test_loss'],
            'final_test_acc': result['final_test_acc'],
            'final_test_loss': result['final_test_loss'],
            'log_dir': str(log_dir)
        }
        
        results.append(run_data)
        
        # Append to CSV immediately
        csv_writer.writerow([
            run_data['run'],
            run_data['best_step'],
            run_data['elapsed_time'],
            run_data['best_test_acc'],
            run_data['best_test_loss'],
            run_data['final_test_acc'],
            run_data['final_test_loss'],
            run_data['log_dir']
        ])
        csv_file.flush()
    
    csv_file.close()

if __name__ == "__main__":
    main()