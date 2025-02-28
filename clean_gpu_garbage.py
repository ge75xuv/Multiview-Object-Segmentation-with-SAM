import os
import gc
import torch

torch.cuda.empty_cache()  # Empties cache
torch.cuda.reset_max_memory_allocated()  # Resets tracking
import gc
gc.collect()
print(torch.cuda.memory_summary(device=None, abbreviated=False))
t = torch.cuda.get_device_properties(0).total_memory
r = torch.cuda.memory_reserved(0)
a = torch.cuda.memory_allocated(0)
print(f'Total Memory: {t}\nReserved Memory: {r}\nAllocated Memory: {a}\n')
device = "cuda"
with torch.cuda.device(device):
    print('Cuda Memory Info:')
    print(f'{torch.cuda.mem_get_info()}\n')  # (free_memory_usage, total_memory)