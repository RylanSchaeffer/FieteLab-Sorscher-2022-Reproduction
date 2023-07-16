import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.utilities import grad_norm
import torch
import torch.nn
import torch.nn.functional
from typing import Any, Callable, Dict, List, Tuple, Union

from src.analyze import compute_ratemaps_2d
from src.ensembles import CellEnsemble, PlaceCellEnsemble, HeadDirectionCellEnsemble
from src.plot import plot_lattice_scores_by_nbins, plot_ratemaps_2d
from src.scores import compute_lattice_scores_from_ratemaps_2d, create_grid_scorers

ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")


class GridCellSystem(pl.LightningModule):
    def __init__(self, wandb_config: Dict, wandb_logger):
        super().__init__()

        # Should save hyperparameters to checkpoint.
        # https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing_basic.html
        # self.save_hyperparameters()

        self.wandb_config = wandb_config
        self.wandb_logger = wandb_logger

        # Sample Place Cell and Head Direction Cells.
        self.pc_ensemble = PlaceCellEnsemble(
            n_cells=self.wandb_config["n_place_cells"],
            stdev=self.wandb_config["place_cell_rf"],
            pos_min=-self.wandb_config["box_width_in_m"] / 2,
            pos_max=self.wandb_config["box_width_in_m"] / 2,
        )

        self.hd_ensemble = HeadDirectionCellEnsemble(
            n_cells=self.wandb_config["n_head_direction_cells"],
            concentration=self.wandb_config["head_direction_cell_concentration"],
        )

        self.recurrent_network = SorscherRecurrentNetwork(wandb_config=wandb_config)

        # TODO: Do we even need this? I think we can delete. This is remnant from Aran.
        # TODO: Check that height and width are in the correct order.
        self.ratemaps_coords_range = (
            (
                -self.wandb_config["box_height_in_m"] / 2.0,
                self.wandb_config["box_height_in_m"] / 2.0,
            ),
            (
                -self.wandb_config["box_width_in_m"] / 2.0,
                self.wandb_config["box_width_in_m"] / 2.0,
            ),
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        # DeepMind's trajectories go a little outside the box.
        # self.log(
        #     f"train/target_pos_min",
        #     batch["target_pos"].min(),
        #     on_step=True,
        #     on_epoch=False,
        #     sync_dist=True,
        # )
        #
        # self.log(
        #     f"train/target_pos_max",
        #     batch["target_pos"].max(),
        #     on_step=True,
        #     on_epoch=False,
        #     sync_dist=True,
        # )

        # assert  >= (-self.wandb_config['box_width_in_m'] / 2.)
        # assert batch['target_pos'].max() <= (self.wandb_config['box_width_in_m'] / 2.)

        init_hd_values, init_pc_or_pos_values, recurrent_inputs = self.compute_inputs(
            batch=batch
        )

        hd_targets, pc_targets = self.compute_hd_and_pc_targets(batch=batch)

        forward_results = self.recurrent_network.forward(
            init_hd_values=init_hd_values,
            init_pc_or_pos_values=init_pc_or_pos_values,
            recurrent_inputs=recurrent_inputs,
        )

        loss_results = self.compute_losses(
            pc_targets=pc_targets,
            pc_logits=forward_results["pc_logits"],
            hd_targets=hd_targets,
            hd_logits=forward_results["hd_logits"],
            pos_targets=batch["target_pos"],
            pos_logits=forward_results["pos_logits"],
        )

        for loss_str, loss_val in loss_results.items():
            self.log(
                f"train/loss={loss_str}",
                loss_val,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )

        pos_decoding_err, pc_acc, hd_acc = self.compute_pos_decoding_err_and_acc(
            pc_logits=forward_results["pc_logits"],
            pc_targets=pc_targets,
            target_pos=batch["target_pos"],
            hd_logits=forward_results["hd_logits"],
            hd_targets=hd_targets,
            pos_logits=forward_results["pos_logits"],
            pos_targets=batch["target_pos"],
        )

        self.log(
            f"train/pos_decoding_err_in_cm",
            100
            * pos_decoding_err,  # Multiplying by 100 converts from meters to centimeters.
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        self.log(f"train/pc_acc", pc_acc, on_step=True, on_epoch=False, sync_dist=True)

        self.log(f"train/pc_acc", hd_acc, on_step=True, on_epoch=False, sync_dist=True)

        return loss_results["total_loss"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        # return
        init_hd_values, init_pc_or_pos_values, recurrent_inputs = self.compute_inputs(
            batch=batch
        )

        hd_targets, pc_targets = self.compute_hd_and_pc_targets(batch=batch)

        forward_results = self.recurrent_network.forward(
            init_hd_values=init_hd_values,
            init_pc_or_pos_values=init_pc_or_pos_values,
            recurrent_inputs=recurrent_inputs,
        )

        loss_results = self.compute_losses(
            pc_targets=pc_targets,
            pc_logits=forward_results["pc_logits"],
            hd_targets=hd_targets,
            hd_logits=forward_results["hd_logits"],
            pos_targets=batch["target_pos"],
            pos_logits=forward_results["pos_logits"],
        )

        for loss_str, loss_val in loss_results.items():
            self.log(
                f"val/loss={loss_str}",
                loss_val,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        pos_decoding_err, pc_acc, hd_acc = self.compute_pos_decoding_err_and_acc(
            pc_logits=forward_results["pc_logits"],
            pc_targets=pc_targets,
            target_pos=batch["target_pos"],
            hd_logits=forward_results["hd_logits"],
            hd_targets=hd_targets,
            pos_logits=forward_results["pos_logits"],
            pos_targets=batch["target_pos"],
        )

        self.log(
            f"val/pos_decoding_err_in_cm",
            100
            * pos_decoding_err,  # Multiplying by 100 converts from meters to centimeters.
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        self.log(f"val/pc_acc", pc_acc, on_step=False, on_epoch=True, sync_dist=True)

        self.log(f"val/hd_acc", pc_acc, on_step=False, on_epoch=True, sync_dist=True)

        positions_numpy = batch["target_pos"].cpu().detach().numpy()
        lstm_activations_numpy = (
            forward_results["lstm_activations"].cpu().detach().numpy()
        )
        g_activations_numpy = forward_results["g_activations"].cpu().detach().numpy()

        for activations_array, activation_str in [
            (lstm_activations_numpy, "lstm"),
            (g_activations_numpy, "g"),
        ]:
            # ratemaps_2d has shape (n_units, nbins_y, nbins_x)
            ratemaps_2d, extreme_coords = compute_ratemaps_2d(
                positions=positions_numpy,
                activations=activations_array,
                coords_range=self.ratemaps_coords_range,
                bin_side_in_m=self.wandb_config["bin_side_in_m"],
            )

            plot_ratemaps_2d(
                ratemaps=ratemaps_2d,
                extreme_coords=extreme_coords,
                wandb_logger=self.wandb_logger,
                wandb_key=f"{activation_str}_ratemaps_2d",
            )

            scorers = create_grid_scorers(
                left=extreme_coords["left"],
                right=extreme_coords["right"],
                top=extreme_coords["top"],
                bottom=extreme_coords["bottom"],
                nbins_list=[ratemaps_2d.shape[1]],  # we're hijacking DeepMind's code.
            )

            lattice_scores_by_nbins_dict = compute_lattice_scores_from_ratemaps_2d(
                ratemaps_2d=ratemaps_2d,
                scorers=scorers,
                n_recurr_units_to_analyze=ratemaps_2d.shape[0],
            )

            quantiles = [0.5, 0.75, 0.9, 0.95, 0.99]
            score_90_quantiles = np.quantile(
                a=lattice_scores_by_nbins_dict[ratemaps_2d.shape[1]][
                    "score_90_by_neuron"
                ],
                q=quantiles,
            )
            for q, s in zip(quantiles, score_90_quantiles):
                self.log(
                    f"val/{activation_str}_score_90_quant={q}",
                    s,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

            plot_lattice_scores_by_nbins(
                lattice_scores_by_nbins_dict=lattice_scores_by_nbins_dict,
                wandb_logger=self.wandb_logger,
                wandb_key=f"{activation_str}_lattice_scores_by_nbins",
            )

    def compute_inputs(self, batch: Dict[str, torch.Tensor]):
        # Create initial conditions.
        with torch.no_grad():
            # Shape: (batch size, 1, n_hd_cells)
            init_hd_values = self.hd_ensemble.get_init(batch["init_hd"])
            if self.wandb_config["hidden_state_init"] == "pc_hd":
                # Shape: (batch size, 1, n_pc_cells)
                init_pc_or_pos_values = self.pc_ensemble.get_init(batch["init_pos"])
            elif self.wandb_config["hidden_state_init"] == "pos_hd":
                # Shape: (batch size, 1, 2)
                init_pc_or_pos_values = batch["init_pos"]
            else:
                raise ValueError(
                    f"Unknown hidden_state_init: {self.wandb_config['hidden_state_init']}"
                )

            if self.wandb_config["input_var"] == "egocentric":
                # Shape: (batch size, 1, 3)
                recurrent_inputs = torch.concat(
                    (batch["ego_speed"], batch["theta_x"], batch["theta_y"]), dim=2
                )
            elif self.wandb_config["input_var"] == "allocentric":
                recurrent_inputs = batch["ego_velocity"]
            else:
                raise ValueError(f"Unknown input_var: {self.wandb_config['input_var']}")

        return init_hd_values, init_pc_or_pos_values, recurrent_inputs

    def compute_losses(
        self,
        pc_targets: torch.Tensor,
        pc_logits: torch.Tensor,
        hd_targets: torch.Tensor,
        hd_logits: torch.Tensor,
        pos_targets: torch.Tensor,
        pos_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Torch cross entropy requires the classes to be in the 1st dimension
        # and also requires the tensors to be contiguous, so we transpose then make contiguous.
        pc_loss = torch.mean(
            ce_loss_fn(
                input=pc_logits.transpose(1, 2).contiguous(),
                target=pc_targets.transpose(1, 2).contiguous(),
            )
        )
        hd_loss = torch.mean(
            ce_loss_fn(
                input=hd_logits.transpose(1, 2).contiguous(),
                target=hd_targets.transpose(1, 2).contiguous(),
            )
        )
        pos_loss = torch.mean(torch.square(pos_targets - pos_logits))

        if self.wandb_config["target_var"] == "pc_hd":
            total_loss = pc_loss + hd_loss
        elif self.wandb_config["target_var"] == "pos":
            total_loss = pos_loss
        else:
            raise ValueError(f"Unknown target_var: {self.wandb_config['target_var']}")
        losses_results = {
            "pc_loss": pc_loss,
            "hd_loss": hd_loss,
            "total_loss": total_loss,
            "pos_loss": pos_loss,
        }
        return losses_results

    def compute_pos_decoding_err_and_acc(
        self,
        pc_logits: torch.Tensor,
        pc_targets: torch.Tensor,
        target_pos: torch.Tensor,
        hd_logits: torch.Tensor,
        hd_targets: torch.Tensor,
        pos_targets: torch.Tensor,
        pos_logits: torch.Tensor,
        num_top_pcs: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.wandb_config["target_var"] == "pc_hd":
            _, top_k_indices = torch.topk(pc_logits, k=num_top_pcs, dim=2)

            # Make sure top_k_indices is of type long, as it's required for indexing
            # top_k_indices = top_k_indices.long()

            # Gathering using advanced indexing
            # Shape: (batch size, sequence length, num_top_pcs, spatial coordinates = 2)
            gathered_means = self.pc_ensemble.means[top_k_indices]

            # Average across the top three PCs.
            # Shape: (batch size, sequence length, 2)
            pred_pos = torch.mean(gathered_means, dim=2)

            pos_decoding_err = torch.mean(
                torch.linalg.norm(pred_pos - target_pos, dim=2)
            )

        elif self.wandb_config["target_var"] == "pos":
            pos_decoding_err = torch.mean(
                torch.linalg.norm(pos_targets - pos_logits, dim=2)
            )
        else:
            raise ValueError(f"Unknown target_var: {self.wandb_config['target_var']}")

        pc_acc = torch.mean(
            torch.eq(
                torch.argmax(pc_logits, dim=2), torch.argmax(pc_targets, dim=2)
            ).float()
        )

        hd_acc = torch.mean(
            torch.eq(
                torch.argmax(hd_logits, dim=2), torch.argmax(hd_targets, dim=2)
            ).float()
        )

        return pos_decoding_err, pc_acc, hd_acc

    def compute_hd_and_pc_targets(self, batch: Dict[str, torch.Tensor]):
        with torch.no_grad():
            target_hd_values = self.hd_ensemble.get_targets(batch["target_hd"])
            target_pc_values = self.pc_ensemble.get_targets(batch["target_pos"])
        return target_hd_values, target_pc_values

    def configure_optimizers(self) -> Dict:
        # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers

        # TODO: Maybe add SWA
        # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.StochasticWeightAveraging.html#pytorch_lightning.callbacks.StochasticWeightAveraging
        if self.wandb_config["optimizer"] == "adadelta":
            optimizer = torch.optim.Adadelta(
                self.parameters(),
                lr=self.wandb_config["learning_rate"],
                weight_decay=self.wandb_config["weight_decay"],
            )
        elif self.wandb_config["optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.wandb_config["learning_rate"],
                weight_decay=self.wandb_config["weight_decay"],
                eps=1e-4,  # https://stackoverflow.com/a/42420014/4570472
            )
        elif self.wandb_config["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.wandb_config["learning_rate"],
                weight_decay=self.wandb_config["weight_decay"],
                eps=1e-4,  # https://stackoverflow.com/a/42420014/4570472
            )
        elif self.wandb_config["optimizer"] == "rmsprop":
            optimizer = torch.optim.RMSprop(
                self.parameters(),
                lr=self.wandb_config["learning_rate"],
                weight_decay=self.wandb_config["weight_decay"],
                momentum=0.9,
                eps=1e-4,
            )
        elif self.wandb_config["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.wandb_config["learning_rate"],
                weight_decay=self.wandb_config["weight_decay"],
                momentum=0.9,
            )
        else:
            # TODO: add adafactor https://pytorch-optimizer.readthedocs.io/en/latest/index.html
            raise NotImplementedError(f"{self.wandb_config['optimizer']}")

        optimizer_and_maybe_others_dict = {
            "optimizer": optimizer,
        }

        if self.wandb_config["learning_rate_scheduler"] is None:
            pass
        elif (
            self.wandb_config["learning_rate_scheduler"]
            == "cosine_annealing_warm_restarts"
        ):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=2,
            )
            optimizer_and_maybe_others_dict["lr_scheduler"] = scheduler

        elif (
            self.wandb_config["learning_rate_scheduler"]
            == "linear_warmup_cosine_annealing"
        ):
            from flash.core.optimizers import LinearWarmupCosineAnnealingLR

            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer=optimizer,
                warmup_epochs=1,
                max_epochs=self.wandb_config["n_epochs"],
            )

            optimizer_and_maybe_others_dict["lr_scheduler"] = scheduler

        elif self.wandb_config["learning_rate_scheduler"] == "reduce_lr_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                factor=0.95,
                optimizer=optimizer,
                patience=3,
            )
            optimizer_and_maybe_others_dict["lr_scheduler"] = scheduler
            optimizer_and_maybe_others_dict["monitor"] = "losses_train/loss"
        else:
            raise NotImplementedError(f"{self.wandb_config['learning_rate_scheduler']}")

        return optimizer_and_maybe_others_dict

    # def on_before_optimizer_step(self, optimizer):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     norms = grad_norm(self.layer, norm_type=2)
    #     self.log_dict(norms)


class SorscherRecurrentNetwork(pl.LightningModule):
    def __init__(self, wandb_config: Dict, **kwargs):
        super().__init__()
        self.wandb_config = wandb_config

        self.pc_or_pos_init_layer = torch.nn.Linear(
            in_features=self.wandb_config["n_place_cells"]
            if self.wandb_config["hidden_state_init"] == "pc_hd"
            else 2,
            out_features=2
            * self.wandb_config["n_hidden_units"],  # 2 is for LSTM state.
            # bias=self.wandb_config['use_bias'],
            bias=False,  # Copied from Sorscher et al. 2022
        )
        self.hd_init_layer = torch.nn.Linear(
            in_features=self.wandb_config["n_head_direction_cells"],
            out_features=2
            * self.wandb_config["n_hidden_units"],  # 2 is for LSTM state.
            # bias=self.wandb_config['use_bias'],
            bias=False,  # Copied from Sorscher et al. 2022
        )
        if self.wandb_config["rnn_type"] == "rnn":
            # self.recurrent_layer = torch.nn.RNN(
            #     input_size=3,
            #     hidden_size=self.wandb_config['n_hidden_units'],
            #     num_layers=1,
            #     nonlinearity='tanh',  # We want squares, not hexagons!
            #     batch_first=True,
            #     bidirectional=False,
            # )
            raise NotImplementedError
        elif self.wandb_config["rnn_type"] == "lstm":
            self.recurrent_layer = torch.nn.LSTM(
                input_size=3 if self.wandb_config["input_var"] == "egocentric" else 2,
                hidden_size=self.wandb_config["n_hidden_units"],
                num_layers=1,
                batch_first=True,
                bidirectional=False,
            )
        else:
            raise ValueError

        self.readout_one_layer = torch.nn.Linear(
            in_features=self.wandb_config["n_hidden_units"],
            out_features=self.wandb_config["n_readout_units"],
            # bias=self.wandb_config['use_bias'],
            bias=True,  # Copied from Sorscher et al. 2022
        )
        self.readout_two_layer = torch.nn.Linear(
            in_features=self.wandb_config["n_readout_units"],
            out_features=self.wandb_config["n_head_direction_cells"]
            + self.wandb_config["n_place_cells"],
            # bias=self.wandb_config['use_bias'],
            bias=True,  # Copied from Sorscher et al. 2022
        )
        assert 0.0 < self.wandb_config["keep_prob"] <= 1.0
        self.dropout_layer = torch.nn.Dropout(
            p=1.0 - self.wandb_config["keep_prob"],
        )
        self.position_layer = torch.nn.Linear(
            in_features=self.wandb_config["n_readout_units"],
            out_features=2,
            bias=self.wandb_config["use_bias"],
        )

    def forward(
        self,
        init_hd_values: torch.Tensor,
        init_pc_or_pos_values: torch.Tensor,
        recurrent_inputs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Compute initial LSTM hidden state values.
        init_hd_activations = self.hd_init_layer(init_hd_values)
        init_pc_or_pos_activations = self.pc_or_pos_init_layer(init_pc_or_pos_values)
        added_init_activations = init_pc_or_pos_activations + init_hd_activations

        # We need to swap batch & time dimension ourselves.
        # We also need to make the tensors contiguous because otherwise get error: rnn: hx is not contiguous
        # https://discuss.pytorch.org/t/runtimeerror-input-is-not-contiguous/930/6
        initial_state = (
            added_init_activations[:, :, : self.wandb_config["n_hidden_units"]]
            .transpose(0, 1)
            .contiguous(),
            added_init_activations[:, :, self.wandb_config["n_hidden_units"] :]
            .transpose(0, 1)
            .contiguous(),
        )

        lstm_activations, _ = self.recurrent_layer(recurrent_inputs, initial_state)
        g_activations = self.readout_one_layer(lstm_activations)
        hd_and_pc_logits = self.readout_two_layer(self.dropout_layer(g_activations))
        pos_logits = self.position_layer(g_activations)

        forward_results = {
            "lstm_activations": lstm_activations,
            "g_activations": g_activations,
            "hd_logits": hd_and_pc_logits[
                :, :, : self.wandb_config["n_head_direction_cells"]
            ],
            "pc_logits": hd_and_pc_logits[
                :, :, self.wandb_config["n_head_direction_cells"] :
            ],
            "pos_logits": pos_logits,
        }

        return forward_results
