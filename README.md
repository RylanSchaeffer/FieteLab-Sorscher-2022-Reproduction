# Fiete Lab Sorscher 2022 Reproduction

## Setup

Create a conda environment with the required packages:

`conda env create --file environment.yml`

To activate the environment:

`conda activate sorscher_reproduction`

Then install wandb with pip following the instructions [here](https://docs.wandb.ai/guides/technical-faq/setup).

`pip install wandb`

## Creating a Sweep

To create a sweep, run the following command at the CLI:

`wandb sweep <path to sweep e.g. sweeps/20230523/basic.yaml>`

This will output the newly created sweep ID:

```
wandb: Creating sweep from: sweeps/20230522/basic.yaml
wandb: Creating sweep with ID: ox6goel1
wandb: View sweep at: https://wandb.ai/rylan/param-sharing/sweeps/ox6goel1
wandb: Run sweep agent with: wandb agent rylan/param-sharing/ox6goel1
```

Then create an "Agent" which will pull the next run off the queue and run it to completion, before repeating.
An agent will continue to function until no more runs in the sweep remain. To create an agent, run:

`wandb agent <your W&B username e.g. rylan>/<project name e.g., param-sharing>/<sweep ID e.g. ox6goel1>`

A sensible way to parallelize is to launch multiple jobs on a cluster, where each job contains 1 agent.
So for instance, if we're running on a SLURM cluster, each SLURM job should create 1 agent.
Rylan has code for this.
