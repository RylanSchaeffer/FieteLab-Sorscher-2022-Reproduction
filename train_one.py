import numpy as np
import json
import os
import pprint
import lightning.pytorch as pl
from lightning.pytorch.callbacks import DeviceStatsMonitor, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import PyTorchProfiler
import shutil
import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from src.globals import default_config
from src.run import set_seed
from src.systems import GridCellSystem
from src.data import TrajectoryDataModule

# Torch settings.
torch.autograd.set_detect_anomaly(True)
# You are using a CUDA device ('A100 80GB PCIe') that has Tensor Cores. To properly utilize them, you
# should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision
# for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
# torch.set_float32_matmul_precision('medium')

# print('CUDA available: ', torch.cuda.is_available())
# print('CUDA device count: ', torch.cuda.device_count())

run = wandb.init(project='sorscher-2022-reproduction',
                 config=default_config)
# Convert to a dictionary; otherwise, can't distribute because W&B
# config is not pickle-able.
wandb_config = dict(wandb.config)

# Convert "None" (type: str) to None (type: NoneType)
for key in ['accumulate_grad_batches', 'gradient_clip_val', 'learning_rate_scheduler']:
    if isinstance(wandb_config[key], str):
        if wandb_config[key] == "None":
            wandb_config[key] = None

# Create checkpoint directory for this run, and save the config to the directory.
run_checkpoint_dir = os.path.join("lightning_logs", wandb.run.id)
os.makedirs(run_checkpoint_dir)
with open(os.path.join(run_checkpoint_dir, 'wandb_config.json'), 'w') as fp:
    json.dump(obj=wandb_config, fp=fp)

# Make sure we set all seeds for maximal reproducibility!
torch_generator = set_seed(seed=wandb_config['seed'])

wandb_logger = WandbLogger(experiment=run)
system = GridCellSystem(wandb_config=wandb_config,
                        wandb_logger=wandb_logger)

# https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.LearningRateMonitor.html
lr_monitor_callback = LearningRateMonitor(
    logging_interval='step',
    log_momentum=True)

checkpoint_callback = ModelCheckpoint(
    monitor='train/loss=total_loss',
    save_top_k=1,
    mode='min',
)

callbacks = [
    lr_monitor_callback,
    checkpoint_callback,
]
# if torch.cuda.is_available():
#     accelerator = 'cuda'
#     devices = torch.cuda.device_count()
#     callbacks.extend([
#         # DeviceStatsMonitor()
#     ])
#     print('GPU available.')
# else:
#     accelerator = 'auto'
#     devices = None
#     callbacks.extend([])
#     print('No GPU available.')

trajectory_datamodule = TrajectoryDataModule(
    wandb_config=wandb_config,
    run_checkpoint_dir=run_checkpoint_dir,
    torch_generator=torch_generator,
)
print('Created Trajectory Datamodule.')

# https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.Trainer.html
trainer = pl.Trainer(
    accelerator='auto',
    accumulate_grad_batches=wandb_config['accumulate_grad_batches'],
    callbacks=callbacks,
    check_val_every_n_epoch=25,
    default_root_dir=run_checkpoint_dir,
    deterministic=True,
    devices='auto',
    # fast_dev_run=True,
    fast_dev_run=False,
    logger=wandb_logger,  # min_epochs=50,
    log_every_n_steps=5,
    # overfit_batches=1,  # useful for debugging
    gradient_clip_val=wandb_config['gradient_clip_val'],
    # gradient_clip_val=None,  # default
    max_epochs=wandb_config['n_epochs'],
    # profiler="simple",  # Simplest profiler
    # profiler="advanced",  # More advanced profiler
    # profiler=PyTorchProfiler(filename=),  # PyTorch specific profiler
    precision=wandb_config['precision'],
    # track_grad_norm=2,
)

# .fit() needs to be called below for multiprocessing.
# See: https://github.com/Lightning-AI/lightning/issues/13039
# See: https://github.com/Lightning-AI/lightning/discussions/9201
# See: https://github.com/Lightning-AI/lightning/discussions/151
if __name__ == '__main__':

    pp = pprint.PrettyPrinter(indent=4)
    print('W&B Config:')
    pp.pprint(wandb_config)

    trainer.fit(
        model=system,
        datamodule=trajectory_datamodule,
    )

    # Delete the data after training finished, to save disk space.
    shutil.rmtree(os.path.join(run_checkpoint_dir, 'data'))
