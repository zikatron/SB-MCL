import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

def plot_algorithm_accuracy(csv_paths: list, column_names: list, algorithm_names: list):
    """
    Reads a list of CSV files, calculates the mean and 95% CI, 
    and plots them as an annotated dot plot.
    """
    
    if not (len(csv_paths) == len(column_names) == len(algorithm_names)):
        print("Error: Input lists must all be the same length.")
        return

    processed_data = []

    # --- 1. Processing Data (Same as before) ---
    for path, col, name in zip(csv_paths, column_names, algorithm_names):
        print(f"Processing: {name}...")
        try:
            df = pd.read_csv(path)
            if col not in df.columns: continue
            data = df[col].dropna()
            if len(data) < 2: continue
        except: continue

        n = len(data)
        mean = np.mean(data)
        sem = stats.sem(data)  
        
        if sem == 0:
            ci_lower, ci_upper = mean, mean
        else:
            try:
                res = stats.bootstrap((data,), np.mean, confidence_level=0.95, method='BCa', random_state=42)
                ci_lower = res.confidence_interval.low
                ci_upper = res.confidence_interval.high
            except:
                ci_lower, ci_upper = stats.t.interval(0.95, n-1, loc=mean, scale=sem)

        processed_data.append({
            "name": name, "mean": mean, 
            "lower_err": mean - ci_lower, 
            "upper_err": ci_upper - mean,
            "ci_low_val": ci_lower,
            "ci_high_val": ci_upper
        })
        
    if not processed_data:
        print("No valid data.")
        return

    # --- 2. Setup Plot Data ---
    plot_names = [d["name"] for d in processed_data]
    plot_means = [d["mean"] for d in processed_data]
    y_errors = [[d["lower_err"] for d in processed_data], [d["upper_err"] for d in processed_data]]
    x_vals = range(len(plot_names))

    fig, ax = plt.subplots(figsize=(10, 7))
    
    # --- 3. Plotting (BLUE DOTS) ---
    ax.errorbar(x_vals, plot_means, yerr=y_errors,
                 fmt='o', color='blue', ecolor='blue', # Blue dots and bars
                 capsize=5, elinewidth=2, markeredgewidth=2,
                 markersize=8, label='Mean Accuracy')

    # --- 4. Annotations (BLACK TEXT, CI ONLY) ---
    # Get current limits to calculate padding
    ylim_bottom, ylim_top = ax.get_ylim()
    y_range = ylim_top - ylim_bottom
    
    for i, d in enumerate(processed_data):
        # Only show the CI Range: [0.955, 0.960]
        label = f"[{d['ci_low_val']:.3f}, {d['ci_high_val']:.3f}]"
        
        # Position text slightly above the error bar
        text_y = d['mean'] + d['upper_err'] + (y_range * 0.02)
        
        ax.text(i, text_y, label, 
                ha='center', va='bottom', 
                fontsize=10, color='black', # Black text
                fontweight='bold')

    # --- 5. Formatting (NO GRID, FIX CLIPPING) ---
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(plot_names, rotation=45, ha='right')
    
    # FIX: Add significant top padding so text doesn't go off-axis
    ax.set_ylim(ylim_bottom, ylim_top + (y_range * 0.25))
    
    # FIX: Add side padding so the text for the first/last items doesn't get cut
    ax.set_xlim(-0.5, len(plot_names) - 0.5)

    # REMOVED: ax.grid() is gone.
    
    plt.tight_layout()
    
    print("\nDisplaying annotated plot...")
    plt.ylim(0, 1.02)
    plt.savefig('algorithm_accuracy_annotated.png', dpi=300, transparent=False)
    plt.show()

def plot_comparison_accuracy_over_time(csv_paths: list, algorithm_names: list, output_filename: str):
    if not (len(csv_paths) == len(algorithm_names)):
        print("Error: Input lists must all be the same length.")
        return

    plt.figure(figsize=(12, 7))
    
    plot_labels = [] 
    x_vals = []

    for csv_path, algorithm_name in zip(csv_paths, algorithm_names):
        print(f"Processing {algorithm_name} from file {csv_path}...")

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  Error reading {csv_path}: {e}. Skipping.")
            continue

        interval_columns = sorted([col for col in df.columns if col.startswith('interval_')], reverse=False)

        if not interval_columns:
            print(f"  Error: No columns found starting with 'interval_' in {csv_path}. Skipping.")
            continue
        
        if not plot_labels:
            raw_ranges = [col.replace('interval_', '') for col in interval_columns]
            raw_ranges.sort(key=lambda s: int(s.split('-')[0]))
            
            new_labels = []
            for r in raw_ranges:
                start, end = map(int, r.split('-'))
                new_labels.append(f"{start + 1}-{end + 1}")
                
            plot_labels = new_labels
            x_vals = range(len(plot_labels))
            
            interval_columns.sort(key=lambda col: int(col.replace('interval_', '').split('-')[0]))
            
            print(f"  Found {len(plot_labels)} class groups: {plot_labels}")

        processed_data = []

        for col_name in interval_columns:
            data = df[col_name].dropna()
            if len(data) < 2:
                processed_data.append(None)
                continue

            n = len(data)
            mean = np.mean(data)
            sem = stats.sem(data)

            if sem == 0:
                ci_lower, ci_upper = mean, mean
            else:
                try:
                    bootstrap_result = stats.bootstrap((data,), np.mean, confidence_level=0.95, method='BCA', random_state=42)
                    ci_lower = bootstrap_result.confidence_interval.low
                    ci_upper = bootstrap_result.confidence_interval.high
                except:
                    df_t = n - 1
                    ci_lower, ci_upper = stats.t.interval(0.95, df_t, loc=mean, scale=sem)

            processed_data.append({
                "mean": mean,
                "lower_err": mean - ci_lower,
                "upper_err": ci_upper - mean
            })

        if not processed_data:
            continue

        plot_means = [d["mean"] if d else np.nan for d in processed_data]
        lower_errors = [d["lower_err"] if d else 0 for d in processed_data]
        upper_errors = [d["upper_err"] if d else 0 for d in processed_data]
        y_errors = [lower_errors, upper_errors]
        
        plt.errorbar(x_vals, plot_means, yerr=y_errors,
                     fmt='o-',
                     capsize=5,
                     markersize=8,
                     label=f'{algorithm_name} (95% CI)')

    if not plot_labels:
        return
        
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Classes', fontsize=12)
    plt.xticks(x_vals, plot_labels, rotation=45, ha='right')
    
    # NO GRIDLINES
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.legend()
    plt.tight_layout()
    plt.ylim(0, 1.02)

    try:
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_filename, dpi=300, transparent=False)
        print(f"\nPlot successfully saved to: {output_filename}")
    except Exception as e:
        print(f"\nError saving plot: {e}")
    plt.close()

def plot_algorithm_violin(csv_paths: list, column_names: list, algorithm_names: list, output_filename='algorithm_violin_comparison.png'):
    if not (len(csv_paths) == len(column_names) == len(algorithm_names)):
        print("Error: Input lists must all be the same length.")
        return

    data_to_plot = []
    clean_names = []
    means = []

    for path, col, name in zip(csv_paths, column_names, algorithm_names):
        try:
            df = pd.read_csv(path)
            if col not in df.columns:
                continue
            raw_data = df[col].dropna().values
            if len(raw_data) < 2:
                continue 
            data_to_plot.append(raw_data)
            clean_names.append(name)
            means.append(np.mean(raw_data))
        except:
            continue

    if not data_to_plot:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    parts = ax.violinplot(data_to_plot, showmeans=False, showmedians=False, showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(0.6)

    inds = np.arange(1, len(data_to_plot) + 1)
    ax.scatter(inds, means, marker='o', color='white', s=30, zorder=3, edgecolor='black', label='Mean')

    for i, dataset in enumerate(data_to_plot):
        min_val, max_val = np.min(dataset), np.max(dataset)
        ax.vlines(inds[i], min_val, max_val, color='k', linestyle='-', lw=1, alpha=0.5)

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_xticks(np.arange(1, len(clean_names) + 1))
    ax.set_xticklabels(clean_names, rotation=45, ha='right')
    
    # NO GRIDLINES
    # ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.ylim(0, 1.02)

    plt.savefig(output_filename, dpi=300, transparent=False)

# --- Execution ---
plot_algorithm_accuracy(
    csv_paths=[
        '/home/bytefuse/batsi/SB-MCL/experiments/gemcl/casia-seq/evaluation-1000t10s-intervals.csv',
        'experiments/oml/casia/evaluation-1000t10s-intervals.csv',
        '/home/bytefuse/batsi/SB-MCL/experiments/std_casia-offline/results.csv',
        '/home/bytefuse/batsi/SB-MCL/experiments/remind-casia-offline/combined_cl_results_detailed.csv'
    ],
    column_names=[
        'overall_accuracy',
        'overall_accuracy',
        'best_test_acc',
        'overall_accuracy'
    ],
    algorithm_names=[
        'GeMCL',
        'OML',
        'Offline',
        'REMIND'
    ]
)

plot_algorithm_violin(
    csv_paths=[
        '/home/bytefuse/batsi/SB-MCL/experiments/gemcl/omniglot-seq/evaluation-1000t10s-intervals.csv',
        'experiments/oml/omniglot/evaluation-1000t10s-intervals.csv',
        '/home/bytefuse/batsi/SB-MCL/experiments/std_omniglot-offline/results.csv',
        '/home/bytefuse/batsi/SB-MCL/experiments/remind-omniglot-offline/combined_cl_results_detailed.csv'
    ],
    column_names=[
        'overall_accuracy',
        'overall_accuracy',
        'best_test_acc',
        'overall_accuracy'
    ],
    algorithm_names=[
        'GeMCL',
        'OML',
        'Offline',
        'REMIND',
    ]
)

plot_comparison_accuracy_over_time(
    csv_paths=[
        '/home/bytefuse/batsi/SB-MCL/experiments/gemcl/casia-seq/evaluation-1000t10s-intervals.csv',
        '/home/bytefuse/batsi/SB-MCL/experiments/oml/casia/evaluation-1000t10s-intervals.csv',
        '/home/bytefuse/batsi/SB-MCL/experiments/remind-casia-offline/combined_cl_results_detailed.csv'
    ],
    algorithm_names=[
        'GeMCL',
        'OML', 
        'REMIND'
    ],
    output_filename='comparison_accuracy_over_time.png'
)