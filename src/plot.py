import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import seaborn as sns
from typing import Dict
import wandb


def plot_lattice_scores_by_nbins(
    lattice_scores_by_nbins_dict: Dict[int, Dict[str, np.ndarray]],
    wandb_logger=None,
    plot_dir: str = None,
    wandb_key: str = None,
):
    plt.close()
    fig, axes = plt.subplots(nrows=1, ncols=len(lattice_scores_by_nbins_dict))

    # If there's only 1 key, axes won't be an array, and thus we can't access into it. This is a workaround.
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for col_idx, (nbins, lattice_scores) in enumerate(
        lattice_scores_by_nbins_dict.items()
    ):
        ax = axes[col_idx]
        ax.hist(lattice_scores["score_60_by_neuron"], bins=30)
        ax.set_title(f"nbins={nbins}")

    plt.tight_layout()

    if plot_dir is not None:
        plot_path = os.path.join(plot_dir, f"lattice_scores_by_nbins.png")
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)

    # https://docs.wandb.ai/guides/track/log/plots#matplotlib-and-plotly-plots
    if wandb_logger is not None:
        wandb_logger.log_image(
            key="lattice_scores_by_nbins" if wandb_key is None else wandb_key,
            images=[wandb.Image(fig)],
        )

    # plt.show()
    plt.close()


def plot_ratemaps_2d(
    ratemaps: np.ndarray,
    wandb_logger=None,
    extreme_coords: Dict[str, float] = None,
    n_rows: int = 10,
    n_cols: int = 10,
    smooth_ratemap_before_plotting: bool = True,
    plot_only_center: bool = True,
    plot_dir: str = None,
    wandb_key: str = None,
):
    plt.close()

    fig, axes = plt.subplots(
        n_rows,  # rows
        n_cols,  # columns
        figsize=(2 * n_rows, 2 * n_cols),
        sharey=True,
        sharex=True,
        gridspec_kw={"width_ratios": [1] * n_cols},
    )

    # If a ratemap is NxN and we only want to plot the center
    # KxK elements, we should take (N-K) / 2.
    # 4 m / 0.05 meters per bin
    # ratemaps has shape (num units, num bins, num bins)
    # center_idx = int(np.floor(ratemaps.shape[1] / (4.0 / 0.05)))

    for unit_idx, ratemap in enumerate(ratemaps):
        if smooth_ratemap_before_plotting:
            ratemap = np.copy(ratemap)
            ratemap[np.isnan(ratemap)] = 0.0
            ratemap = scipy.ndimage.gaussian_filter(ratemap, sigma=2.0)

        row, col = unit_idx // n_cols, unit_idx % n_cols
        ax = axes[row, col]

        # if plot_only_center:
        #     array_to_plot = ratemap[center_idx:-center_idx, center_idx:-center_idx]
        # else:
        #     array_to_plot = ratemap

        # sns.heatmap(
        #     data=ratemap,
        #     ax=ax,
        #     cbar=False,
        #     cmap='Spectral_r',
        #     square=True,
        #     yticklabels=False,
        #     xticklabels=False)

        # Seaborn's heatmap flips the y-axis by default. Flip it back ourselves.
        # ax.invert_yaxis()

        ax.imshow(ratemap, cmap="Spectral_r", interpolation="none")

        ax.set_title(
            "Unit: {}\nMax: {:.3f}\nMin: {:.3f}".format(
                unit_idx,
                np.max(ratemap),
                np.min(ratemap),
            )
        )

        if extreme_coords is not None:
            if col == 0:
                ax.set_ylabel(
                    "({:.1f}, {:.1f})".format(
                        extreme_coords["bottom"],
                        extreme_coords["top"],
                    )
                )
            if row == 0:
                ax.set_xlabel(
                    "({:.1f}, {:.1f})".format(
                        extreme_coords["left"],
                        extreme_coords["right"],
                    )
                )

        if unit_idx == (n_rows * n_cols - 1):
            break

    plt.tight_layout()
    # plot_path = os.path.join(plot_dir, f'ratemaps_{epoch_idx}.png')
    # plt.savefig(plot_path,
    #             bbox_inches='tight',
    #             dpi=300)
    # plt.show()

    if wandb_logger is not None:
        # https://docs.wandb.ai/guides/track/log/plots#matplotlib-and-plotly-plots
        fig = plt.gcf()
        wandb_logger.log_image(
            key="ratemaps" if wandb_key is None else wandb_key,
            images=[wandb.Image(fig)],
        )

    plt.close()


def save_plot_with_multiple_extensions(plot_dir: str, plot_title: str):
    # Ensure that axis labels don't overlap.
    plt.gcf().tight_layout()

    extensions = [
        # "pdf",
        "png",
    ]
    for extension in extensions:
        plot_path = os.path.join(plot_dir, plot_title + f".{extension}")
        print(f"Plotted {plot_path}")
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
