import torch

# 1) Version & basic CUDA availability
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# 2) Device info
idx = 0
print("  Device #0 name:      ", torch.cuda.get_device_name(idx))
print("  Compute capability:  ", torch.cuda.get_device_capability(idx))

# 3) Supported architectures (nightly → get_arch_list; stable may not list all)
if hasattr(torch.cuda, "get_arch_list"):
    print("  Arch list:", torch.cuda.get_arch_list())

# 4) A simple tensor op on GPU
x = torch.randn(4, 4, device="cuda")
y = x * 3.14
print("  GPU tensor op ok:", y)
