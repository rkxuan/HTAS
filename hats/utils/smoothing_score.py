import os
import numpy as np
from typing import List
import matplotlib.pyplot as plt

def plot_smoothing_training_score(
    utilities_pos: List[float], 
    utilities_neg: List[float], 
    folder_path:str,
    file_name:str,
    window_size: int = 10,
):
    assert window_size > 1, "please set window_size > 1"
    file_path = os.path.join(folder_path, file_name)
    utils_pos_np = np.array(utilities_pos)
    x_pos = np.arange(1, len(utils_pos_np) + 1)
    effective_window_pos = min(window_size, len(utils_pos_np))
    weights_pos = np.ones(effective_window_pos) / effective_window_pos
    smoothed_utils_pos = np.convolve(utils_pos_np, weights_pos, mode='valid')
    x_smoothed_pos = x_pos[(effective_window_pos - 1):]

    utils_neg_np = np.array(utilities_neg)
    x_neg = np.arange(1, len(utils_neg_np) + 1)
    effective_window_neg = min(window_size, len(utils_neg_np))
    weights_neg = np.ones(effective_window_neg) / effective_window_neg
    smoothed_utils_neg = np.convolve(utils_neg_np, weights_neg, mode='valid')
    x_smoothed_neg = x_neg[(effective_window_neg - 1):]

    plt.figure(figsize=(10, 8))
    plt.plot(x_smoothed_neg, smoothed_utils_neg, label=f'With negative samples', linewidth=2)
    plt.plot(x_smoothed_pos, smoothed_utils_pos, label=f'Without negative samples', linewidth=2)

    plt.xlabel('The number of training samples', fontsize=16)
    plt.ylabel('Training Score', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)
    plt.savefig(file_path)
