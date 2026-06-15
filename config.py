# config.py
from dotenv import load_dotenv
import os
import torch

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
    