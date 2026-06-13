# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "psutil>=7.2.2",
# ]
# ///
import os
import psutil

def get_current_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)
