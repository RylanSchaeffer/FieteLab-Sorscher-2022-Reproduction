# Fiete Lab Sorscher 2022 Reproduction

This repository contains a subset of code contained withour our recent preprint
Disentangling Fact from Grid Cell Fiction in Trained Deep Path Integrators.

![](figures/sorscher_rebuttal.png)

As background, our NeurIPS 2022 paper [No free lunch from deep learning in neuroscience: A case study through models of the entorhinal-hippocampal circuit](https://proceedings.neurips.cc/paper_files/paper/2022/file/66808849a9f5d8e2d00dbdc844de6333-Paper-Conference.pdf)
was unable to reproduce key findings of Sorscher et al. NeurIPS 2019 [A unified theory for the origin of grid cells through
the lens of pattern formation](https://ganguli-gang.stanford.edu/pdf/19.DecodePattern.pdf).
After the NeurIPS 2022 review process, [code subsequently released](https://github.com/ganguli-lab/grid-pattern-formation/blob/master/square_grid_cells.ipynb)
by Sorscher et al.; however, it was implemented in a deprecated version of TensorFlow (1.14.0),
so we reproduced it here using PyTorch and PyTorch Lightning, then integrated W&Bs to sweep hyperparameters and 
log results for further analysis.

## Setup

Create a conda environment with the required packages:

`conda env create --file environment.yml`

To activate the environment:

`conda activate sorscher_reproduction`

Upgrade pip:

`pip install --upgrade pip`

Then install wandb with pip, recommended by W&B [here](https://docs.wandb.ai/guides/technical-faq/setup).

`pip install wandb`

Then install pytorch lightning:

`pip install lightning`

Then install the correct version of Pytorch:

`pip install torch==1.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html`

I also needed to install an additional system component via conda:

`conda install libgcc`

`pip3 install torch torchvision torchaudio`

## Running

### Running a Single Run

To run a single experiment, run the following command:

`python train_one.py`

The hyperparameters for this run are specified in `src/globals.py`. The results should
automatically be logged to W&B.

### Running via W&B Sweep

To create a sweep, run the following command at the CLI:

`wandb sweep <path to sweep e.g. sweeps/20230523/basic.yaml>`

This will output the newly created sweep ID, e.g.,:

```
wandb: Creating sweep from: sweeps/20230522/basic.yaml
wandb: Creating sweep with ID: ox6goel1
wandb: View sweep at: https://wandb.ai/rylan/sorscher-2022-reproduction/sweeps/ox6goel1
wandb: Run sweep agent with: wandb agent rylan/sorscher-2022-reproduction/ox6goel1
```

Then create an "Agent" which will pull the next run off the queue and run it to completion, before repeating.
An agent will continue to function until no more runs in the sweep remain. To create an agent, run:

`wandb agent <your W&B username e.g. rylan>/sorscher-2022-reproduction/<sweep ID e.g. ox6goel1>`

A sensible way to parallelize is to launch multiple jobs on a cluster, where each job contains 1 agent.
So for instance, if we're running on a SLURM cluster, each SLURM job should create 1 agent.
Rylan has code for this.

## Analysis

For analysis code, see `notebooks/`.

## Contact

Questions? Comments? Interested in collaborating? Open an issue or email rylanschaeffer@gmail.com.