import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PRED_INDEX = list(range(1, 13))

INPUT_LEN = 300
