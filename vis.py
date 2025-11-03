import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from dataset import DATASET
from train import get_config, prepare_data # Assuming these are in your local files
import matplotlib.patches as patches
import yaml # PyYAML is needed for loading .yaml config files


# =============================================================================

num_samples = 6
num_tasks = 3

def visualize_data_tasks():
    """
    Visualize 6 training images grouped into 3 distinct tasks,
    each in a labeled box.
    """
    # 1. We now want exactly 6 samples to form 3 tasks
    
    # Set seed for reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Load config (adjust path to your config file)
    config_path = './cfg/casia-std.yaml'  # Change to your config
    config = get_config(config_path)
    print(f"Loaded config from: {config_path}")
    print(f"Dataset: {config['dataset']}")
    
    # Use EXACT same data loading as train_std.py
    Dataset = DATASET[config['dataset']]
    meta_test_set = Dataset(config, root='./data', meta_split='test')
    meta_test_loader = DataLoader(
        meta_test_set,
        batch_size=config['eval_batch_size'],
        num_workers=0,
        collate_fn=meta_test_set.collate_fn)
    
    # Get one task's data
    train_x, train_y, _, _ = next(iter(meta_test_loader))
    
    # Create train_set
    train_set = meta_test_set.get_tensor_dataset(train_x[0], train_y[0])
    
    # Create train_loader with shuffle=False to get consistent samples
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=False)
    
    # Get first batch from train_loader
    train_x_batch, train_y_batch = next(iter(train_loader))
    


    # Take the first 6 samples for our 3 tasks
    images = train_x_batch[:num_samples].cpu()

    # # Print actual character names for each sample using y_dict mapping
    # if hasattr(meta_test_set, 'y_dict'):
    #     id_to_char = {v: k for k, v in meta_test_set.y_dict.items()}
    #     print("Sample character names:")
    #     for idx in range(num_samples):
    #         class_idx = train_y_batch[idx].item()
    #         char_name = id_to_char.get(class_idx, str(class_idx))
    #         print(f"Sample {idx+1}: Class {class_idx} - Character '{char_name}'")

    # Create the figure. The figsize is kept the same.
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 4.5))

    # Plot all 6 images first
    for idx in range(num_samples):
        img = images[idx].numpy()

        if img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))

        if img.shape[-1] == 1:
            img = img.squeeze(-1)

        cmap = 'gray' if len(img.shape) == 2 else None
        axes[idx].imshow(img, cmap=cmap)
        axes[idx].axis('off')

        if idx % 2 == 0:
            axes[idx].set_title('Class 1', fontsize=12, y=-0.35)
        else:
            axes[idx].set_title('Class 2', fontsize=12, y=-0.35)

    # --- KEY CHANGE HERE ---
    # Remove tight_layout and manually adjust subplot spacing.
    # wspace adds horizontal space between the subplots.
    plt.subplots_adjust(wspace=0.4, top=0.85, bottom=0.18)

    # Add boxes and headings for each task
    for i in range(num_tasks):
        ax1_idx = i * 2
        ax2_idx = i * 2 + 1

        pos1 = axes[ax1_idx].get_position()
        pos2 = axes[ax2_idx].get_position()

        # Define the bounding box dimensions with some padding
        x0 = pos1.x0 - 0.015
        y0 = pos1.y0 - 0.15 # Adjusted for new layout
        width = pos2.x1 - pos1.x0 + 0.03
        height = pos1.height + 0.25 # Adjusted for new layout

        rect = patches.Rectangle(
            (x0, y0), width, height,
            linewidth=2,
            edgecolor='black',
            facecolor='none',
            transform=fig.transFigure,
            clip_on=False
        )

        fig.add_artist(rect)

        fig.text(
            x0 + width / 2,
            y0 + height, # Positioned exactly at the top of the box
            f'Task {i+1}',
            ha='center',
            va='bottom', # Anchor text from its bottom edge
            fontsize=14,
            fontweight='bold',
            color='black',
            transform=fig.transFigure
        )

    # Save the final plot
    save_path = f'task_visualization_separated_seed{SEED}.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight') # Removed bbox_inches for more consistent saving
    print(f'\n✓ Visualization with separated tasks saved to: {save_path}')

    plt.show()

# =============================================================================


if __name__ == '__main__':
    try:
        visualize_data_tasks()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure that 'dataset.py', './cfg/casia-std.yaml', and the './data' directory are correctly set up.")