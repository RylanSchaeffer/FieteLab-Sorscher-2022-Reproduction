import numpy as np
import json
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import DeviceStatsMonitor, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler
import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from ssl_gc.globals import default_config
from ssl_gc.run import load_pretrained_model, set_seed
from ssl_gc.systems import SSLGridCellSystem
from ssl_gc.trajectory_generator import TrajectoryDataModule

# Torch settings.
torch.autograd.set_detect_anomaly(True)
# You are using a CUDA device ('A100 80GB PCIe') that has Tensor Cores. To properly utilize them, you
# should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision
# for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
torch.set_float32_matmul_precision('medium')

run = wandb.init(project='sorscher-2022-reproduction',
                 config=default_config)
# Convert to a dictionary; otherwise, can't distribute because W&B
# config is not pickle-able.
wandb_config = dict(wandb.config)

# Convert "None" (type: str) to None (type: NoneType)
for key in ['accumulate_grad_batches', 'auto_scale_batch_size', 'gradient_clip_val',
            'learning_rate_scheduler', 'mask_type']:
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
system = SSLGridCellSystem(wandb_config=wandb_config,
                           wandb_logger=wandb_logger)


# https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.LearningRateMonitor.html
lr_monitor_callback = LearningRateMonitor(
    logging_interval='step',
    log_momentum=True)

checkpoint_callback = ModelCheckpoint(
    monitor='losses_train/loss',
    save_top_k=1,
    mode='min',
)

callbacks = [
    lr_monitor_callback,
    checkpoint_callback,
]
if torch.cuda.is_available():
    accelerator = 'gpu'
    devices = torch.cuda.device_count()
    callbacks.extend([
        # DeviceStatsMonitor()
    ])
    print('GPU available.')
else:
    accelerator = None
    devices = None
    callbacks.extend([])
    print('No GPU available.')

trajectory_datamodule = TrajectoryDataModule(
    wandb_config=wandb_config,
    batch_size=wandb_config['batch_size'],
    torch_generator=torch_generator,
)


# https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.Trainer.html
trainer = pl.Trainer(
    accelerator=accelerator,
    accumulate_grad_batches=wandb_config['accumulate_grad_batches'],
    auto_lr_find=wandb_config['auto_lr_find'],
    auto_scale_batch_size=False,
    # auto_scale_batch_size=wandb_config['auto_scale_batch_size'],
    callbacks=callbacks,
    check_val_every_n_epoch=3,  # default
    default_root_dir=run_checkpoint_dir,
    deterministic=True,
    devices=devices,
    logger=wandb_logger,  # min_epochs=50,
    log_every_n_steps=500,
    # overfit_batches=1,  # useful for debugging
    gradient_clip_val=wandb_config['gradient_clip_val'],
    # gradient_clip_val=None,  # default
    max_epochs=wandb_config['n_epochs'],
    # profiler="simple",  # Simplest profiler
    # profiler="advanced",  # More advanced profiler
    # profiler=PyTorchProfiler(filename=),  # PyTorch specific profiler
    precision=wandb_config['precision'],
    track_grad_norm=2,
)

# .fit() needs to be called below for multiprocessing.
# See: https://github.com/Lightning-AI/lightning/issues/13039
# See: https://github.com/Lightning-AI/lightning/discussions/9201
# See: https://github.com/Lightning-AI/lightning/discussions/151
if __name__ == '__main__':

    # If we want to autoscale learning rate or batch size, we need to call .tune() first.
    if wandb_config['auto_lr_find'] or wandb_config['auto_scale_batch_size'] is not None:
        # trainer.tune(
        #     model=system,
        #     datamodule=trajectory_datamodule,
        #     scale_batch_size_kwargs={'max_trials': 25}  # default is 25
        # )
        print('Reminder: tuning learning rate or batch size is not yet implemented.')
    else:
        print('Not tuning learning rate or batch size.')

    trainer.fit(
        model=system,
        datamodule=trajectory_datamodule,
    )
