import torch
import logging
import numpy as np
import numpy.typing as npt

from src.logFormatter import logFormatter

def pretty_time_delta(seconds) -> str:
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return '%dd %dh %dm %ds' % (days, hours, minutes, seconds)
    elif hours > 0:
        return '%dh %dm %ds' % (hours, minutes, seconds)
    elif minutes > 0:
        return '%dm %ds' % (minutes, seconds)
    else:
        return '%ds' % (seconds,)


def is_number(x):

    try:
        float(x)

    except ValueError:
        return False

    return True

def otsu_threshold(arr: npt.NDArray) -> float:

    num_bins = 256
    hist, bin_edges = np.histogram(arr, bins=num_bins, range=(0,1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Normalize histogram to probabilities
    probabilities = hist / arr.size

    optimal_threshold = 0.0
    max_between_class_variance = 0.0

    for t in range(1, num_bins):
        # Class probabilities (weights)
        weight_background = np.sum(probabilities[:t])
        weight_foreground = np.sum(probabilities[t:])

        if weight_background == 0 or weight_foreground == 0:
            continue

        # Class means
        mean_background = np.sum(bin_centers[:t] * probabilities[:t]) / weight_background
        mean_foreground = np.sum(bin_centers[t:] * probabilities[t:]) / weight_foreground

        # Calculate between-class variance
        between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        # Check if this is the best threshold so far
        if between_class_variance > max_between_class_variance:
            max_between_class_variance = between_class_variance
            optimal_threshold = bin_centers[t]

    return optimal_threshold


def setup_logger():
    log_colors_dict = {
        'DEBUG': 'grey',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red'
    }

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logFormatter(log_colors_dict))

    logger.addHandler(ch)

def calculate_metrics(y_hat: torch.Tensor, y: torch.Tensor, threshold: float=0.5) -> tuple[float, float, float]:

    #optimal_threshold = otsu_threshold(y_hat)
    #print(optimal_threshold)
    #y_hat = torch.where(y_hat >= threshold, 1, 0)

    TP = torch.sum(y_hat == y)
    FP = torch.sum(y_hat == y)
    TN = torch.sum(y_hat == y)
    FN = torch.sum(y_hat == y)

    #print(f'true positive: {TP}, true negative {TN}, false positive: {FP}, false negative: {FN}')

    # Calculate metrics

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return accuracy, precision, recall

def calculate_guessing_entropy(y_hat: torch.Tensor, y: torch.Tensor) -> float:

    _, sorted_indices = torch.sort(y_hat, descending=True, dim=1)
    ranks = (sorted_indices == y.unsqueeze(1)).nonzero(as_tuple=True)[1]
    return (ranks.float() + 1).mean()