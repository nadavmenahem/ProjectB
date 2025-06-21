import torch
# Check if CUDA (NVIDIA GPU support) is available
print("CUDA available:", torch.cuda.is_available())

# If available, how many GPUs?
print("Number of GPUs:", torch.cuda.device_count())

# And the name of each GPU
for i in range(torch.cuda.device_count()):
    print(f" GPU {i}: {torch.cuda.get_device_name(i)}")

# Example of selecting the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
