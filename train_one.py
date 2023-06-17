# This is going to train a single network.
import json
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from theory_of_polysemanticity.data import PolysemanticityDataModule
from theory_of_polysemanticity.globals import default_run_config
from theory_of_polysemanticity.systems import PolysemanticitySystem
from theory_of_polysemanticity.utils import set_seed

wandb_run = wandb.init(project='sorscher_2022_reproduction',
                       config=default_run_config)
wandb_run_config = dict(wandb_run.config)


run_checkpoint_dir = os.path.join("lightning_logs", wandb_run.id)
os.makedirs(run_checkpoint_dir)
with open(os.path.join(run_checkpoint_dir, 'wandb_config.json'), 'w') as fp:
    json.dump(obj=wandb_run_config, fp=fp)


# Make sure this run is reproducible.
set_seed(seed=wandb_run_config['seed'])

# See example at:
# https://github.com/wandb/examples/blob/master/colabs/pytorch-lightning/Fine_tuning_a_Transformer_with_Pytorch_Lightning.ipynb
wandb_logger = WandbLogger(experiment=wandb_run)

# Construct the network.
grid_cell_system = PolysemanticitySystem(
    wandb_run_config=wandb_run_config,
    wandb_logger=wandb_logger,
)

# Construct the dataset and dataloader.
polysemanticity_datamodule = PolysemanticityDataModule(
    wandb_run_config=wandb_run_config,
)

# Train the network.
trainer = pl.Trainer(
    # accumulate_grad_batches
    deterministic=True,
    logger=wandb_logger,
    log_every_n_steps=100,
    precision=wandb_run_config['precision'],
)
trainer.fit(model=polysemanticity_system,
            datamodule=polysemanticity_datamodule)
print('Finished training one network!')
