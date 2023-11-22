import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

import src.analyze
import src.plot

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=True,
)


runs_configs_and_results_df = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="sorscher-2022-reproduction",
    data_dir=data_dir,
    sweep_ids=[
        "jgobd84j",  # Default optimizer: Adam
        "vbi49msz",  # Optimizers
    ],
    refresh=False,
    user="rylan",
)


# Reshape the results dataframe for easy plotting.
grid_score_percentile_columns = [
    "g/score_90_quant=0.99",
    "g/score_90_quant=0.95",
    "g/score_90_quant=0.9",
    "g/score_90_quant=0.75",
    "g/score_90_quant=0.5",
]
other_columns = ["optimizer", "seed", "train/pos_decoding_err_in_cm"]

position_decoding_err_and_grid_scores_df = runs_configs_and_results_df[
    other_columns + grid_score_percentile_columns
]

position_decoding_err_and_grid_scores_melted_df = (
    position_decoding_err_and_grid_scores_df.melt(
        id_vars=other_columns, var_name="grid_score_percentile", value_name="grid_score"
    )
)
position_decoding_err_and_grid_scores_melted_df[
    "grid_score_percentile"
] = position_decoding_err_and_grid_scores_melted_df.apply(
    lambda row: float(row["grid_score_percentile"].split("=")[1]), axis=1
)

# RMSProp doesn't achieve low path decoding error but also doesn't learn lattices,
# so it makes the grid plots look bad. To be as generous as possible, we exclude it.
position_decoding_err_and_grid_scores_melted_df = (
    position_decoding_err_and_grid_scores_melted_df[
        position_decoding_err_and_grid_scores_melted_df["optimizer"] != "rmsprop"
    ]
)

plt.close()
g = sns.lineplot(
    data=position_decoding_err_and_grid_scores_melted_df,
    x="grid_score_percentile",
    y="grid_score",
    hue="optimizer",
)
plt.xlabel("Optimizer")
plt.ylabel(r"$90^{\circ}$ Grid Score")
g.legend_.set_title("Grid Score Distribution\nPercentile")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir, plot_title="score_90_vs_optimizer_by_grid_score_percentile"
)
plt.show()


plt.close()
g = sns.barplot(
    data=position_decoding_err_and_grid_scores_melted_df,
    x="optimizer",
    y="train/pos_decoding_err_in_cm",
)
plt.xlabel("Optimizer")
plt.ylabel("Position Decoding Error (cm)")
g.set(ylim=(0, 110))
# Add dotted black horizontal line at 15 cm.
g.axhline(y=15, color="black", linestyle="--")
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir, plot_title="pos_decoding_err_vs_optimizer"
)
plt.show()

print("Finished 02_optimizers!")
