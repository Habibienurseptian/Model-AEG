import os
import torch


def optimize_cpu():
    num_threads = os.cpu_count()

    try:
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(max(1, num_threads // 2))
    except RuntimeError:
        print("PyTorch thread settings already initialized, skipping...")

    print("CPU cores:", num_threads)
    print("PyTorch intra-op threads:", torch.get_num_threads())
    print("PyTorch inter-op threads:", torch.get_num_interop_threads())