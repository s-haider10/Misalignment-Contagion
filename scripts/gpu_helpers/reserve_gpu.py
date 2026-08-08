import torch
import time
import sys

gpu_id = int(sys.argv[1])
torch.cuda.set_device(gpu_id)
# Allocate ~20GB to reserve the GPU
x = torch.zeros((1024, 1024, 1024, 2, 5), dtype=torch.uint8, device=f'cuda:{gpu_id}')
print(f"Reserved GPU {gpu_id}, allocated {x.element_size() * x.nelement() / 1e9:.2f} GB", flush=True)
while True:
    time.sleep(60)
