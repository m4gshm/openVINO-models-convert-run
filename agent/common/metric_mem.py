"""Process memory metric used to log model loading footprint."""
import os
import psutil

def get_current_memory():
    """Return resident set memory of this process in megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)
