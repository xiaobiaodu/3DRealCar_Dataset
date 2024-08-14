import torch
import psutil

def get_cpu_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage = process.memory_info().rss / (1024 ** 3)
    return memory_usage

def get_gpu_memory():
    return (torch.cuda.memory_allocated() + torch.cuda.memory_reserved()) / (1024 ** 3)
