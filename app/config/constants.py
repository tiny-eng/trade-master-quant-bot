import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PRED_INDEX = [1, 5, 15, 30, 60]

INPUT_LEN = 60 * 5
