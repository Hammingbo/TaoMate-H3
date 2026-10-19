"""Launch with torchrun --standalone --nproc-per-node=8 to verify NCCL."""
import datetime
import os

import torch
import torch.distributed as dist

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
try:
    value = torch.tensor([dist.get_rank() + 1.0], device="cuda")
    dist.all_reduce(value)
    assert value.item() == 36.0, value.item()
    if dist.get_rank() == 0:
        print("NCCL_8_GPU_OK", flush=True)
finally:
    dist.destroy_process_group()
