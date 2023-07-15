import numpy as np
import os
import pandas as pd
import scipy.stats
from typing import Dict, List, Tuple
import wandb


def compute_ratemaps_2d(
    positions: np.ndarray,
    activations: np.ndarray,
    coords_range: Tuple[Tuple[float, float], Tuple[float, float]],
    bin_side_in_m: float = 0.05,
    statistic: str = "mean",
) -> Tuple[np.ndarray, Dict[str, float]]:
    """

    :param positions: Shape: (batch size, traj len, 2)
    :param activations: Shape: (batch size, traj len, num units)
    :param bin_side_in_m:
    :param statistic:
    :param coords_range:
    :return:
    """

    n_rnn_units = activations.shape[2]

    xs = positions[:, :, 0].flatten()
    ys = positions[:, :, 1].flatten()
    # n_bins = int((coords_range[0][1] - coords_range[0][0]) / bin_side_in_m)

    extreme_coords = {
        "left": np.min(xs),
        "right": np.max(xs),
        "top": np.max(ys),
        "bottom": np.min(ys),
    }

    most_positive_coord = max(extreme_coords["right"], extreme_coords["top"])
    most_negative_coord = min(extreme_coords["left"], extreme_coords["bottom"])
    span = most_positive_coord - most_negative_coord
    n_bins = int(span / bin_side_in_m) + 1

    ratemaps = np.zeros(shape=(n_rnn_units, n_bins, n_bins), dtype=np.float32)

    for unit_idx in range(activations.shape[2]):
        ratemaps[unit_idx] = scipy.stats.binned_statistic_2d(
            xs,
            ys,
            activations[:, :, unit_idx].flatten(),
            bins=n_bins,
            statistic=statistic,
            range=coords_range,
        )[0]

    return ratemaps, extreme_coords
