import subprocess
import os

# --- Configuration ---

# The script name to run
SCRIPT_NAME = "train_remind.py" 

# The base path to your experiments
# This is the parent folder containing seed_1, seed_2, etc.
BASE_EXP_DIR = "/home/bytefuse/batsi/SB-MCL/experiments/remind-casia-offline"

# Total number of seeds to run
NUM_SEEDS = 20

# --- End Configuration ---

print(f"Starting experiment runner for {NUM_SEEDS} seeds...")

for seed_num in range(1, NUM_SEEDS + 1):
    print(f"\n--- Running Seed {seed_num}/{NUM_SEEDS} ---")
    
    # 1. Construct the dynamic paths for this seed
    log_dir = os.path.join(BASE_EXP_DIR, f"seed_{seed_num}")
    config_file = os.path.join(log_dir, "config.yaml")
    
    # Check if paths exist
    if not os.path.isdir(log_dir):
        print(f"Warning: Directory not found, skipping: {log_dir}")
        continue
    if not os.path.isfile(config_file):
        print(f"Warning: Config file not found, skipping: {config_file}")
        continue
        
    # 2. Build the command
    command = [
        "python",
        SCRIPT_NAME,
        "--config", config_file,
        "--log-dir", log_dir,
        "--run_number", str(seed_num)
    ]
    
    print(f"Executing: {' '.join(command)}")
    
    # 3. Run the command and wait for it to complete
    try:
        subprocess.run(command, check=True)
        print(f"--- Finished Seed {seed_num} ---")
    except subprocess.CalledProcessError as e:
        print(f"!!! ERROR running seed {seed_num}: {e} !!!")
        print("Stopping runner.")
        break
    except KeyboardInterrupt:
        print("\nRunner stopped by user.")
        break

print("\nAll runs complete.")