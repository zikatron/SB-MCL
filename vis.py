import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from dataset import DATASET
from train import get_config, prepare_data 
import matplotlib.patches as patches
import yaml 

# =============================================================================

num_samples = 6
num_tasks = 3

def visualize_data_tasks():
    """
    Visualize 6 training images grouped into 3 distinct tasks.
    Labels are forced to:
    Row 1: Class 1 / Class 2
    Row 2: Character 1 ... Character 6
    """
    
    # Set seed for reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Load config
    config_path = './cfg/casia-std.yaml'
    config = get_config(config_path)
    print(f"Loaded config from: {config_path}")
    
    # Data Loading
    Dataset = DATASET[config['dataset']]
    meta_test_set = Dataset(config, root='./data', meta_split='test')
    meta_test_loader = DataLoader(
        meta_test_set,
        batch_size=config['eval_batch_size'],
        num_workers=0,
        collate_fn=meta_test_set.collate_fn)
    
    train_x, train_y, _, _ = next(iter(meta_test_loader))
    train_set = meta_test_set.get_tensor_dataset(train_x[0], train_y[0])
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=False)
    
    # Get batch
    train_x_batch, train_y_batch = next(iter(train_loader))
    images = train_x_batch[:num_samples].cpu()

    # Create the figure
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 4.5))

    # Plot all 6 images
    for idx in range(num_samples):
        img = images[idx].numpy()

        if img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))

        if img.shape[-1] == 1:
            img = img.squeeze(-1)

        cmap = 'gray' if len(img.shape) == 2 else None
        axes[idx].imshow(img, cmap=cmap)
        axes[idx].axis('off')

        # --- MODIFIED LABELING LOGIC ---
        
        # 1. Top Label: Alternates Class 1 / Class 2
        if idx % 2 == 0:
            class_label = "Class 1"
        else:
            class_label = "Class 2"
            
        # 2. Bottom Label: Sequential Character 1 to 6
        # We force this to ensure the figure matches your text description
        char_label = f"Character {idx + 1}"

        # 3. Set Title with newline
        axes[idx].set_title(f"{class_label}\n({char_label})", fontsize=11, y=-0.45)

    # Adjust spacing to make room for the taller labels
    plt.subplots_adjust(wspace=0.4, top=0.85, bottom=0.22) 

    # Add boxes and headings for each task
    for i in range(num_tasks):
        ax1_idx = i * 2
        ax2_idx = i * 2 + 1

        pos1 = axes[ax1_idx].get_position()
        pos2 = axes[ax2_idx].get_position()

        # Box dimensions
        x0 = pos1.x0 - 0.015
        y0 = pos1.y0 - 0.18 # Lowered to cover the new text
        width = pos2.x1 - pos1.x0 + 0.03
        height = pos1.height + 0.28 # Increased height for new text

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
            y0 + height,
            f'Task {i+1}',
            ha='center',
            va='bottom',
            fontsize=14,
            fontweight='bold',
            color='black',
            transform=fig.transFigure
        )

    # Save
    save_path = f'task_visualization_chars_seed{SEED}.png'
    plt.savefig(save_path, dpi=200) 
    print(f'\n✓ Visualization saved to: {save_path}')

    plt.show()

if __name__ == '__main__':
    try:
        visualize_data_tasks()
    except Exception as e:
        print(f"\nAn error occurred: {e}")