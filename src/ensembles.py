import numpy as np
import lightning.pytorch as pl
import torch
import torch.distributions
import torch.nn


class CellEnsemble(pl.LightningModule):
    """Abstract parent class for place and head direction cell ensembles."""

    def __init__(
        self, n_cells, soft_targets: str = "softmax", soft_init: str = "softmax"
    ):
        super().__init__()
        self.n_cells = n_cells
        if soft_targets not in ["softmax", "voronoi", "sample", "normalized"]:
            raise ValueError
        else:
            self.soft_targets = soft_targets
        # Provide initialization of LSTM in the same way as targets if not specified
        # i.e one-hot if targets are Voronoi
        if soft_init is None:
            self.soft_init = soft_targets
        else:
            if soft_init not in ["softmax", "voronoi", "sample", "normalized", "zeros"]:
                raise ValueError
            else:
                self.soft_init = soft_init

    def get_targets(self, x: torch.Tensor):
        """Type of target."""

        if self.soft_targets == "normalized":
            targets = torch.exp(self.unnor_logpdf(x))
        elif self.soft_targets == "softmax":
            lp = self.log_posterior(x)
            targets = softmax(lp)
        elif self.soft_targets == "sample":
            lp = self.log_posterior(x)
            targets = softmax_sample(lp)
        elif self.soft_targets == "voronoi":
            lp = self.log_posterior(x)
            targets = one_hot_max(lp)
        return targets

    def get_init(self, x):
        """Type of initialisation."""

        if self.soft_init == "normalized":
            init = torch.exp(self.unnor_logpdf(x))
        elif self.soft_init == "softmax":
            lp = self.log_posterior(x)
            init = softmax(lp)
        elif self.soft_init == "sample":
            lp = self.log_posterior(x)
            init = softmax_sample(lp)
        elif self.soft_init == "voronoi":
            lp = self.log_posterior(x)
            init = one_hot_max(lp)
        elif self.soft_init == "zeros":
            init = torch.zeros_like(self.unnor_logpdf(x))
        return init

    def loss(self, predictions, targets):
        """Loss."""

        if self.soft_targets == "normalized":
            smoothing = 1e-2
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=(1.0 - smoothing) * targets + smoothing * 0.5,
                logits=predictions,
                name="ensemble_loss",
            )
            loss = torch.mean(loss, dim=-1)
        else:
            loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=targets, logits=predictions, name="ensemble_loss"
            )
        return loss

    def log_posterior(self, x):
        logp = self.unnor_logpdf(x)
        log_posteriors = logp - torch.logsumexp(logp, dim=2, keepdim=True)
        return log_posteriors


class PlaceCellEnsemble(CellEnsemble, pl.LightningModule):
    """Calculates the dist over place cells given an absolute position."""

    def __init__(
        self,
        n_cells: int,
        stdev: float = 0.35,
        pos_min: float = -5.0,
        pos_max: float = 5,
        normalization: str = "lse",
        soft_targets: str = "softmax",
        soft_init: str = "softmax",
    ):
        super(PlaceCellEnsemble, self).__init__(
            n_cells=n_cells, soft_targets=soft_targets, soft_init=soft_init
        )
        # Create a random MoG with fixed cov over the position (Nx2)
        # This need to be a parameter, otherwise Lightning won't automatically move to GPU.
        self.means = torch.nn.Parameter(
            (pos_max - pos_min) * torch.rand(size=(self.n_cells, 2), device=self.device)
            + pos_min,
            requires_grad=False,
        )
        assert -1.1 <= self.means.min()
        assert self.means.max() <= 1.1
        # self.variances = torch.nn.Parameter(
        #     torch.ones((self.n_cells, 2), device=self.device) * stdev * stdev,
        #     requires_grad=False)
        self.variances = stdev * stdev
        assert normalization in {"lse", "lse_no_grad", "none", "max"}
        self.normalization = normalization

    def unnor_logpdf(
        self,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # Output the probability of each component at each point (BxTxN)  # Shape: (batch, time, n_cells, 2)
        diff = positions[:, :, np.newaxis, :] - self.means[np.newaxis, np.newaxis, ...]
        unnor_logp = (
            -0.5 * torch.square(torch.linalg.norm(diff, dim=-1)) / self.variances
        )
        return unnor_logp

    def log_posterior(self, x: torch.Tensor):
        logp = self.unnor_logpdf(x)
        if self.normalization == "none":
            log_posteriors = logp
        elif self.normalization == "lse":
            log_posteriors = logp - torch.logsumexp(logp, dim=2, keepdim=True)
        elif self.normalization == "lse_no_grad":
            log_posteriors = logp - torch.logsumexp(logp, dim=2, keepdim=True).detach()
        elif self.normalization == "max_no_grad":
            log_posteriors = logp - torch.max(logp, dim=2, keepdim=True).detach()
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")
        return log_posteriors


class HeadDirectionCellEnsemble(CellEnsemble, pl.LightningModule):
    """Calculates the dist over HD cells given an absolute angle."""

    def __init__(
        self,
        n_cells: int,
        concentration: float = 20.0,
        soft_targets: str = "softmax",
        soft_init: str = "softmax",
    ):
        super(HeadDirectionCellEnsemble, self).__init__(
            n_cells=n_cells, soft_targets=soft_targets, soft_init=soft_init
        )
        # Create a random Von Mises with fixed precision over the angles.
        # Need to make these parameters so they're moved onto the GPU by Lightning!
        self.means = torch.nn.Parameter(
            2.0 * np.pi * torch.rand(n_cells, device=self.device) - np.pi,
            requires_grad=False,
        )
        # self.kappa = torch.nn.Parameter(concentration * torch.ones(n_cells, device=self.device),
        #                                 requires_grad=False)
        self.kappa = concentration

    def unnor_logpdf(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # Shape: (batch, time, dim)
        return self.kappa * torch.cos(x - self.means[np.newaxis, np.newaxis, :])


def softmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x."""
    return torch.softmax(x, dim=axis)


def softmax_sample(x):
    """Sample the categorical distribution from logits and sample it."""
    dist = torch.distributions.Categorical(logits=x)
    return dist.sample()
