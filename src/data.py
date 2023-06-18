import numpy as np
import os
import pytorch_lightning as pl
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage.filters import uniform_filter1d
import torch
from torch.utils.data import Dataset, IterableDataset
from typing import Dict, Generator, List, Tuple
from torch.utils.data import Dataset, DataLoader
from typing import Generator, List, Tuple


class TrajectoryDataset(Dataset):

    def __init__(self,
                 wandb_config: Dict,
                 run_checkpoint_dir: str,
                 split: str = 'train',
                 ):

        self.wandb_config = wandb_config
        self.dataset_dir = os.path.join(run_checkpoint_dir, 'data', split)
        self.split = split
        if self.split == 'train':
            self.length = self.wandb_config['batch_size'] * self.wandb_config['n_batches_per_epoch']
        elif self.split == 'val':
            self.length = self.wandb_config['batch_size']
        else:
            raise ValueError('Invalid split: {}'.format(self.split))

        self.generated_data = None
        if self.wandb_config['data_generation'] == 'sampled_at_beginning':
            # Sample all the data now and then write to disk.
            os.makedirs(self.dataset_dir)
            for i in range(self.length):
                generated_data = generate_trajectory(
                    batch_size=1,
                    sequence_length=self.wandb_config['sequence_length'],
                    dt=self.wandb_config['dt'],
                    box_size=self.wandb_config['box_width_in_m'],
                )
                # Remove the batch dimension.
                generated_data = {k: v[0] for k, v in generated_data.items()}
                np.savez(file=os.path.join(self.dataset_dir, f'{i}'),
                        **generated_data)

        elif self.wandb_config['data_generation'] == 'sampled_at_every_step':
            pass
        else:
            raise ValueError('Invalid data generation type.')

    def __len__(self):
        return self.length  # the size of the dataset

    def __getitem__(self,
                    idx: int) -> Dict[str, np.ndarray]:

        if self.wandb_config['data_generation'] == 'sampled_at_beginning':
            npzfile = np.load(os.path.join(self.dataset_dir, f'{idx}.npz'))
            item = {k: v.astype(np.float32) for k, v in npzfile.items()}

        elif self.wandb_config['data_generation'] == 'sampled_at_every_step':
            # Sample the data now.
            generated_data = generate_trajectory(
                batch_size=1,
                sequence_length=self.wandb_config['sequence_length'],
                dt=self.wandb_config['dt'],
                box_size=self.wandb_config['box_width_in_m'],
            )
            item = {k: v[0].astype(np.float32) for k, v in generated_data.items()}

        else:
            raise ValueError('Invalid data generation type.')

        return item


class TrajectoryDataModule(pl.LightningDataModule):

    def __init__(self,
                 wandb_config: Dict,
                 run_checkpoint_dir: str,
                 torch_generator: torch.Generator = None,
                 num_workers: int = None):
        super().__init__()
        self.wandb_config = wandb_config
        self.run_checkpoint_dir = run_checkpoint_dir  # Necessary if we write data to disk.
        self.torch_generator = torch_generator

        # Recommendation: https://pytorch-lightning.readthedocs.io/en/latest/guides/speed.html
        if num_workers is None:
            num_workers = max(4, os.cpu_count() // 4)  # heuristic
            # num_workers = 1
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None

        # Allegedly pinning memory saves time.
        self.pin_memory = torch.cuda.is_available()

    def setup(self, stage: str):

        self.train_dataset = TrajectoryDataset(
            wandb_config=self.wandb_config,
            run_checkpoint_dir=self.run_checkpoint_dir,
            split='train')

        self.val_dataset = TrajectoryDataset(
            wandb_config=self.wandb_config,
            run_checkpoint_dir=self.run_checkpoint_dir,
            split='val')

        print(f'TrajectoryDataModule.setup(stage={stage}) called.')

    def train_dataloader(self):
        print('TrajectoryDataModule.train_dataloader() called.')
        return DataLoader(
            self.train_dataset,
            batch_size=self.wandb_config['batch_size_train'],
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=self.pin_memory,
            # worker_init_fn=self.worker_init_fn,
            generator=self.torch_generator,
        )

    def val_dataloader(self):
        print('TrajectoryDataModule.val_dataloader() called.')
        return DataLoader(
            self.val_dataset,
            batch_size=self.wandb_config['batch_size_val'],
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # worker_init_fn=self.worker_init_fn,
            generator=self.torch_generator,
        )

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished.
        print(f'TrajectoryDataModule.teardown(stage={stage}) called.')


def min_dist_angle(position, direction, box_size):
    x, y = position
    dists = [box_size - x, box_size - y, box_size + x, box_size + y]
    d_wall = np.min(dists)
    angles = np.arange(4) * np.pi / 2
    theta = angles[np.argmin(dists)]
    a_wall = direction - theta
    a_wall = (a_wall + np.pi) % (2 * np.pi) - np.pi
    return d_wall, a_wall


def generate_trajectory(batch_size: int,
                        sequence_length: int,
                        dt: float,
                        b: float = 0.13 * 2 * np.pi,  # forward velocity rayleigh dist scale (m/sec),
                        border_region: float = 0.03,  # meters
                        box_size: float = 1.1,  # meters
                        mu: float = 0.0,
                        sigma: float = 5.76 * 2  # stdev rotation velocity (rads/sec)
                        ):

    # initialize variables
    position = np.zeros((batch_size, sequence_length + 1, 2), dtype=float)
    head_dir = np.zeros((batch_size, sequence_length + 1), dtype=float)
    turning = np.zeros((batch_size, sequence_length + 1), dtype='bool')
    ego_speed = np.zeros((batch_size, sequence_length + 1), dtype=float)

    for batch_idx in range(batch_size):

        position[batch_idx, 0] = np.random.uniform(-box_size, box_size, 2)
        head_dir[batch_idx, 0] = np.random.uniform(0, 2 * np.pi)

        # generate sequence of random boosts and turns
        random_turn = np.random.normal(mu, sigma, sequence_length + 1)
        random_vel = np.random.rayleigh(b, sequence_length + 1)

        v = np.random.rayleigh(b)
        for i in range(1, sequence_length + 1):
            # If in border region, turn and slow down
            d_wall, a_wall = min_dist_angle(position[batch_idx, i - 1], head_dir[batch_idx, i - 1] % (2 * np.pi), box_size)
            if d_wall < border_region and np.abs(a_wall) < np.pi / 2:
                turning[batch_idx, i - 1] = 1
                turn_angle = np.sign(a_wall) * (np.pi / 2 - np.abs(a_wall)) + dt * random_turn[i]
                v = 0.25 * v
            else:
                v = random_vel[i]
                turn_angle = dt * random_turn[i]
            # Take a step
            ego_speed[batch_idx, i - 1] = v * dt
            position[batch_idx, i] = position[batch_idx, i - 1] + ego_speed[batch_idx, i - 1] * \
                                     np.asarray([np.cos(head_dir[batch_idx, i - 1]), np.sin(head_dir[batch_idx, i - 1])])
            # Rotate head direction
            head_dir[batch_idx, i] = head_dir[batch_idx, i - 1] + turn_angle

    ang_velocity = np.diff(head_dir, axis=1)
    theta_x, theta_y = np.cos(ang_velocity), np.sin(ang_velocity)
    head_dir = (head_dir + np.pi) % (2 * np.pi) - np.pi  # Constrain head_dir to interval (-pi, pi)

    # All arrays have shape (batch size = 1, appropriate temporal length, dim of variable)
    generated_data = {
        'init_pos': position[:, 0, np.newaxis, :],
        'init_hd': head_dir[:, 0, np.newaxis, np.newaxis],
        'ego_speed': ego_speed[:, :-1, np.newaxis],
        'theta_x': theta_x[:, :, np.newaxis],
        'theta_y': theta_y[:, :, np.newaxis],
        'target_pos': position[:, 1:, :],
        'target_hd': head_dir[:, 1:, np.newaxis]
    }
    return generated_data
