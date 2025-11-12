import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

def calculate_mean(file_path: str, column_name: str) -> float:
    """
    Calculate the mean of a specified column in a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.
    column_name (str): The name of the column to calculate the mean for.

    Returns:
    float: The mean of the specified column.
    """
    data = pd.read_csv(file_path)
    mean_value = data[column_name].mean()
    return mean_value

def plot_algorithm_accuracy(csv_paths: list, column_names: list, algorithm_names: list):
    """
    Reads a list of CSV files, calculates the mean and 95% CI for a
    specified column in each, and plots the results as an error-bar dot plot.

    Args:
        csv_paths (list): A list of file paths to the CSV files.
        column_names (list): A parallel list of column names to use for
                             calculating stats from each CSV.
        algorithm_names (list): A parallel list of names for the x-axis.
    """
    
    if not (len(csv_paths) == len(column_names) == len(algorithm_names)):
        print("Error: Input lists must all be the same length.")
        return

    processed_data = []

    # Iterate over all three lists simultaneously
    for path, col, name in zip(csv_paths, column_names, algorithm_names):
        print(f"Processing: {name} (File: {path}, Column: {col})...")
        
        # --- 1. Read Data ---
        try:
            df = pd.read_csv(path)
            if col not in df.columns:
                print(f"  Warning: Column '{col}' not found in '{path}'. Skipping.")
                continue
                
            # Get data, drop any missing values
            data = df[col].dropna()

            if len(data) < 2:
                print(f"  Warning: Not enough data (found {len(data)}) in '{col}' from '{path}'. Skipping.")
                continue

        except FileNotFoundError:
            print(f"  Error: File not found at '{path}'. Skipping.")
            continue
        except Exception as e:
            print(f"  Error reading {path}: {e}. Skipping.")
            continue

        # --- 2. Calculate Stats & 95% CI ---
        n = len(data)
        mean = np.mean(data)
        sem = stats.sem(data)  # Standard Error of the Mean
        
        if sem == 0:
            # All values are identical, CI is just the mean
            ci_lower, ci_upper = mean, mean
            print("  Note: All data points are identical.")
        else:
            # Calculate the 95% CI using bootstrapping
            # We bootstrap the mean (np.mean)
            # (data,) is the required format for the data argument (a sequence of samples)
            # method='BCA' (Bias-Corrected and Accelerated) is a robust default
            # random_state=42 ensures reproducibility of the bootstrap
            
            # The bootstrap function can be slow on large datasets, but is more robust
            print("  Calculating bootstrapped CI (this may take a moment)...")
            try:
                bootstrap_result = stats.bootstrap((data,), 
                                                   np.mean, 
                                                   confidence_level=0.95, 
                                                   method='BCa', 
                                                   random_state=42)
                
                ci_lower = bootstrap_result.confidence_interval.low
                ci_upper = bootstrap_result.confidence_interval.high
            except Exception as e:
                print(f"  Error during bootstrapping: {e}. Falling back to t-interval.")
                # Fallback to t-interval if bootstrapping fails
                df = n - 1
                confidence = 0.95
                ci_lower, ci_upper = stats.t.interval(confidence, df, loc=mean, scale=sem)


        # --- 3. Store for Plotting ---
        # plt.errorbar needs the error *size* relative to the mean
        lower_error = mean - ci_lower
        upper_error = ci_upper - mean
        
        processed_data.append({
            "name": name,
            "mean": mean,
            "lower_err": lower_error,
            "upper_err": upper_error
        })
        
        print(f"  Mean: {mean:.4f}, 95% CI: ({ci_lower:.4f}, {ci_upper:.4f})")

    # --- 4. Plot Results ---
    if not processed_data:
        print("No valid data was processed. No plot will be generated.")
        return

    # Unzip the processed data into lists for plotting
    plot_names = [d["name"] for d in processed_data]
    plot_means = [d["mean"] for d in processed_data]
    lower_errors = [d["lower_err"] for d in processed_data]
    upper_errors = [d["upper_err"] for d in processed_data]
    
    # y_errors must be in the format [[lower_errors_list], [upper_errors_list]]
    y_errors = [lower_errors, upper_errors]
    
    # X-axis values are just the indices
    x_vals = range(len(plot_names))

    plt.figure(figsize=(10, 7))
    
    # Create the error bar plot
    plt.errorbar(x_vals, plot_means, yerr=y_errors,
                 fmt='o',         # 'o' = dot at the mean
                 capsize=5,     # Adds the end caps to the error bars
                 linestyle='None',# Don't connect the dots
                 markersize=8,
                 label='Mean Accuracy (95% CI)')

    # --- 5. Format Plot ---
    plt.title('Algorithm Accuracy Comparison', fontsize=16)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Algorithm', fontsize=12)
    
    # Set the x-axis ticks and labels
    plt.xticks(x_vals, plot_names, rotation=45, ha='right')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout() # Adjusts plot to prevent labels from overlapping
    
    print("\nDisplaying plot...")
    plt.savefig('algorithm_accuracy_comparison.png')

def plot_comparison_accuracy_over_time(csv_paths: list, algorithm_names: list, output_filename: str):
    """
    Reads a list of CSV files, each containing accuracy data across different
    intervals for a different algorithm. Plots all algorithms on a single
    line graph with bootstrapped 95% CIs and saves the plot to a file.

    The function expects all CSVs to share the same 'interval_...' columns.

    Args:
        csv_paths (list): A list of file paths to the CSV files.
        algorithm_names (list): A parallel list of names for each algorithm (for the legend).
        output_filename (str): The full path and filename to save the plot (e.g., './plots/comparison.png').
    """
    
    if not (len(csv_paths) == len(algorithm_names)):
        print("Error: Input lists must all be the same length.")
        return

    plt.figure(figsize=(12, 7))
    
    plot_labels = [] # To store x-axis labels (e.g., '0-99')
    x_vals = []

    # --- 1. Iterate over each algorithm/CSV ---
    for csv_path, algorithm_name in zip(csv_paths, algorithm_names):
        print(f"Processing {algorithm_name} from file {csv_path}...")

        # --- 2. Read Data ---
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"  Error: File not found at '{csv_path}'. Skipping.")
            continue
        except Exception as e:
            print(f"  Error reading {csv_path}: {e}. Skipping.")
            continue

        # --- 3. Find Interval Columns ---
        # Find all columns that start with 'interval_'
        interval_columns = sorted([col for col in df.columns if col.startswith('interval_')])

        if not interval_columns:
            print(f"  Error: No columns found starting with 'interval_' in {csv_path}. Skipping {algorithm_name}.")
            continue
        
        # On the first successful run, set the x-axis labels
        if not plot_labels:
            plot_labels = [col.replace('interval_', '') for col in interval_columns]
            x_vals = range(len(plot_labels))
            print(f"  Found {len(plot_labels)} intervals. Using as x-axis.")

        processed_data = []

        # --- 4. Process each interval column for this algorithm ---
        for col_name in interval_columns:
            data = df[col_name].dropna()

            if len(data) < 2:
                print(f"  Warning: Not enough data for '{col_name}'. Skipping.")
                processed_data.append(None) # Add placeholder to maintain index
                continue

            n = len(data)
            mean = np.mean(data)
            sem = stats.sem(data)

            if sem == 0:
                ci_lower, ci_upper = mean, mean
                print(f"  {col_name}: All values identical.")
            else:
                # Calculate bootstrapped CI
                print(f"  Calculating bootstrapped CI for {col_name}...")
                try:
                    bootstrap_result = stats.bootstrap((data,), 
                                                       np.mean, 
                                                       confidence_level=0.95, 
                                                       method='BCA', 
                                                       random_state=42)
                    
                    ci_lower = bootstrap_result.confidence_interval.low
                    ci_upper = bootstrap_result.confidence_interval.high
                except Exception as e:
                    print(f"  Error during bootstrapping for {col_name}: {e}. Falling back to t-interval.")
                    df_t = n - 1
                    ci_lower, ci_upper = stats.t.interval(0.95, df_t, loc=mean, scale=sem)

            # Store for plotting
            lower_error = mean - ci_lower
            upper_error = ci_upper - mean
            
            processed_data.append({
                "mean": mean,
                "lower_err": lower_error,
                "upper_err": upper_error
            })
            
            print(f"    Mean: {mean:.4f}, 95% CI: ({ci_lower:.4f}, {ci_upper:.4f})")

        # --- 5. Plot this algorithm's line ---
        if not processed_data:
            print(f"No valid data processed for {algorithm_name}.")
            continue

        # Filter out any 'None' placeholders if intervals were skipped
        plot_means = [d["mean"] if d else np.nan for d in processed_data]
        lower_errors = [d["lower_err"] if d else 0 for d in processed_data]
        upper_errors = [d["upper_err"] if d else 0 for d in processed_data]
        y_errors = [lower_errors, upper_errors]
        
        plt.errorbar(x_vals, plot_means, yerr=y_errors,
                     fmt='o-',        # 'o-' = dot + connecting line
                     capsize=5,
                     markersize=8,
                     label=f'{algorithm_name} (95% CI)')

    # --- 6. Format Final Plot ---
    if not plot_labels:
        print("No data was successfully plotted. No output file generated.")
        plt.close()
        return
        
    # plt.title('Algorithm Accuracy Over Time Comparison', fontsize=16)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('classes Interval', fontsize=12)
    
    plt.xticks(x_vals, plot_labels, rotation=45, ha='right')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # --- 7. Save Plot ---
    try:
        # Ensure directory exists if output_filename includes a path
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        plt.savefig(output_filename, dpi=300)
        print(f"\nPlot successfully saved to: {output_filename}")
    except Exception as e:
        print(f"\nError saving plot: {e}")
        
    plt.close() # Close the figure to free memory

plot_algorithm_accuracy(
    csv_paths=[
        '/home/bytefuse/batsi/SB-MCL/experiments/gemcl/casia/evaluation-1000t10s-intervals.csv',
        'experiments/oml/casia/evaluation-1000t10s-intervals.csv',
        '/home/bytefuse/batsi/SB-MCL/experiments/STD-offline_epoch_fixed_lr/casia/results.csv',
        '/home/bytefuse/batsi/SB-MCL/results.csv'
    ],
    column_names=[
        'overall_accuracy',
        'overall_accuracy',
        'best_test_acc',
        'best_test_acc'
    ],
    algorithm_names=[
        'GEMCL',
        'OML',
        'Offline-10-shots',
        'Offline-50-shots'
    ]
)

plot_comparison_accuracy_over_time(
    csv_paths=[
        '/home/bytefuse/batsi/SB-MCL/experiments/gemcl/omniglot/evaluation-1000t10s-intervals.csv',
        '/home/bytefuse/batsi/SB-MCL/experiments/oml/evaluation-1000t10s-intervals.csv',
    ],
    algorithm_names=[
        'GEMCL',
        'OML', 

    ],
    output_filename='comparison_accuracy_over_time.png'
)