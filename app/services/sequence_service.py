import numpy as np

INPUT_LEN = 300

def create_input_sequence(data_scaled):
    X = data_scaled[-INPUT_LEN:]
    return X.reshape(1, INPUT_LEN, 1)
