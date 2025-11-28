import torch

file_path = "/home/bytefuse/batsi/SB-MCL/experiments/gemcl/evaluation-10t10s-meta_test.pt"

# Load the content
loaded_data = torch.load(file_path)

# You can then inspect or use the loaded_data
print(loaded_data)