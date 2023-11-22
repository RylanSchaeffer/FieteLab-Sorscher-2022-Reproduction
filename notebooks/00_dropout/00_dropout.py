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
        "yrspxhq4",  # Dropout
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
other_columns = ["keep_prob", "seed", "train/pos_decoding_err_in_cm"]

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
position_decoding_err_and_grid_scores_melted_df["dropout_probability"] = (
    1.0 - position_decoding_err_and_grid_scores_melted_df["keep_prob"]
)

plt.close()
g = sns.lineplot(
    data=position_decoding_err_and_grid_scores_melted_df,
    x="dropout_probability",
    y="grid_score",
    hue="grid_score_percentile",
)
plt.xlabel("Dropout Probability")
plt.ylabel(r"$90^{\circ}$ Grid Score")
g.set(xscale="log")
g.legend_.set_title("Grid Score Percentile")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir, plot_title="score_90_vs_keep_prob_by_grid_score_percentile"
)
plt.show()

print("Finished 00_dropout!")
