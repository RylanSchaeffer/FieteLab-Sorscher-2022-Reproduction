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


def download_wandb_project_runs_configs(
    wandb_project_path: str,
    data_dir: str,
    sweep_ids: List[str] = None,
    finished_only: bool = True,
    refresh: bool = False,
    user: str = "rylan",
) -> pd.DataFrame:
    runs_configs_df_path = os.path.join(
        data_dir, "sweeps=" + ",".join(sweep_ids) + "_runs_configs.csv"
    )
    if refresh or not os.path.isfile(runs_configs_df_path):
        # Download sweep results
        api = wandb.Api(timeout=100)

        # Project is specified by <entity/project-name>
        if sweep_ids is None:
            runs = api.runs(path=wandb_project_path)
        else:
            runs = []
            for sweep_id in sweep_ids:
                # TODO: add custom path
                sweep = api.sweep(f"{user}/{wandb_project_path}/{sweep_id}")
                runs.extend([run for run in sweep.runs])
                # runs.extend(api.runs(path=wandb_project_path,
                #                      filters={"Sweep": sweep_id}))

        sweep_results_list = []
        for run in runs:
            # .summary contains the output keys/values for metrics like accuracy.
            #  We call ._json_dict to omit large files
            summary = run.summary._json_dict

            # .config contains the hyperparameters.
            #  We remove special values that start with _.
            summary.update(
                {k: v for k, v in run.config.items() if not k.startswith("_")}
            )

            summary.update(
                {
                    "State": run.state,
                    "Sweep": run.sweep.id if run.sweep is not None else None,
                    "run_id": run.id,
                }
            )
            # .name is the human-readable name of the run.
            summary.update({"run_name": run.name})
            sweep_results_list.append(summary)

        runs_configs_df = pd.DataFrame(sweep_results_list)

        # Save to disk.
        runs_configs_df.to_csv(runs_configs_df_path, index=False)
        print(f"Wrote {runs_configs_df_path} to disk.")
    else:
        runs_configs_df = pd.read_csv(runs_configs_df_path)
        print(f"Loaded {runs_configs_df_path} from disk.")

    # Keep only finished runs
    finished_runs = runs_configs_df["State"] == "finished"
    print(
        f"% of successfully finished runs: {finished_runs.mean()} ({finished_runs.sum()} / {len(finished_runs)})"
    )

    if finished_only:
        runs_configs_df = runs_configs_df[finished_runs]

        # Check that we don't have an empty data frame.
        assert len(runs_configs_df) > 0

        # Ensure we aren't working with a slice.
        runs_configs_df = runs_configs_df.copy()

    return runs_configs_df


def download_wandb_project_runs_histories(
    wandb_project_path: str,
    data_dir: str,
    sweep_ids: List[str] = None,
    num_samples: int = 10000,
    refresh: bool = False,
    keys: List[str] = None,
    user: str = "rylan",
) -> pd.DataFrame:
    if keys is None:
        keys = [
            "epoch",
            "val/percent_units_active_each_datum_mean",
            "val/percent_units_active_each_datum_std",
        ]

    runs_histories_df_path = os.path.join(
        data_dir, "sweeps=" + ",".join(sweep_ids) + "_runs_histories.csv"
    )
    if refresh or not os.path.isfile(runs_histories_df_path):
        # Download sweep results
        api = wandb.Api(timeout=60)

        # Project is specified by <entity/project-name>
        if sweep_ids is None:
            runs = api.runs(path=wandb_project_path)
        else:
            runs = []
            for sweep_id in sweep_ids:
                # TODO: change project lookup to take entire directory name instead
                sweep = api.sweep(f"{user}/{wandb_project_path}/{sweep_id}")
                runs.extend([run for run in sweep.runs])
                # runs.extend(api.runs(path=wandb_project_path,
                #                      filters={"Sweep": sweep_id}))

        runs_histories_list = []
        for run in runs:
            history = run.history()
            if history.empty:
                continue
            run_history = run.scan_history(
                # samples=num_samples + ,
                keys=keys
                + ["_step"]
            )
            run_history_df = pd.DataFrame(run_history)
            run_history_df["run_id"] = run.id
            runs_histories_list.append(run_history_df)

        assert len(runs_histories_list) > 0
        runs_histories_df = pd.concat(runs_histories_list, sort=False)

        # runs_histories_df.sort_values(["run_id", "_step"], ascending=True, inplace=True)

        runs_histories_df.reset_index(inplace=True, drop=True)
        runs_histories_df.to_csv(runs_histories_df_path, index=False)
        print(f"Wrote {runs_histories_df_path} to disk")
    else:
        runs_histories_df = pd.read_csv(runs_histories_df_path)
        print(f"Loaded {runs_histories_df_path} from disk.")

    return runs_histories_df


def setup_notebook_dir(
    notebook_dir: str,
    refresh: bool = False,
) -> Tuple[str, str]:
    # Declare paths.
    data_dir = os.path.join(notebook_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    results_dir = os.path.join(notebook_dir, "results")
    if refresh:
        import shutil

        if os.path.exists(results_dir) and os.path.isdir(results_dir):
            shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    return data_dir, results_dir
