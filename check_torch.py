import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"CUDA version: {torch.version.cuda}")

# Try to move a tensor to GPU
try:
    x = torch.randn(1).cuda()
    print("Tensor to GPU: SUCCESS")
except Exception as e:
    print(f"Tensor to GPU: FAILED - {e}")
