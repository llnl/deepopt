"""
Shared output scaling utilities for DeepOpt models.
"""
from typing import Any, Dict, Optional

import torch


class StandardizeOutputScaler:
    """
    Legacy z-score output scaler for old BoTorch Standardize checkpoints.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-12) -> None:
        self.mean = mean.float().detach()
        std = std.float().detach()
        self.std = torch.where(std.abs() >= eps, std, torch.ones_like(std))
        self.eps = eps

    def to(self, device: torch.device) -> "StandardizeOutputScaler":
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def transform(self, Y: torch.Tensor, X: Optional[torch.Tensor] = None) -> torch.Tensor:
        mean, std = self._standardization_tensors(Y)
        return (Y - mean) / std

    def inverse_transform(self, Y: torch.Tensor, X: Optional[torch.Tensor] = None) -> torch.Tensor:
        mean, std = self._standardization_tensors(Y)
        return Y * std + mean

    def inverse_variance(self, Yvar: torch.Tensor, X: Optional[torch.Tensor] = None) -> torch.Tensor:
        _, std = self._standardization_tensors(Yvar)
        return Yvar * std.pow(2)

    def inverse_covariance(self, covariance: torch.Tensor, X: Optional[torch.Tensor] = None) -> torch.Tensor:
        std = self.std.to(covariance.device).squeeze()
        if std.numel() == 1:
            return covariance * std.pow(2)
        return covariance * std.unsqueeze(-1) * std.unsqueeze(-2)

    @classmethod
    def from_botorch_state_dict(
        cls,
        state: Dict[str, torch.Tensor],
        device: Optional[torch.device] = None,
        prefix: str = "outcome_transform.",
    ) -> "StandardizeOutputScaler":
        mean_key = f"{prefix}means"
        std_key = f"{prefix}stdvs"
        std_sq_key = f"{prefix}_stdvs_sq"
        if mean_key not in state:
            raise RuntimeError("Legacy Standardize checkpoint is missing outcome_transform.means.")
        if std_key in state:
            std = state[std_key]
        elif std_sq_key in state:
            std = state[std_sq_key].sqrt()
        else:
            raise RuntimeError(
                "Legacy Standardize checkpoint is missing standard deviation statistics and cannot be loaded."
            )
        scaler = cls(state[mean_key], std)
        if device is not None:
            scaler.to(device)
        return scaler

    def _standardization_tensors(self, Y: torch.Tensor) -> tuple:
        return self.mean.to(Y.device), self.std.to(Y.device)


class OutputScaler:
    """
    Min/max output scaler with optional per-fidelity scaling.

    Parameters
    ----------
    multi_fidelity : bool, default=False
        If True, fit one output scale per fidelity level using the last input
        column as the fidelity index.
    num_fidelities : int, default=1
        Number of fidelity levels.
    fidelity_dim : int, default=-1
        Input dimension containing fidelity indices.
    eps : float, default=1e-12
        Minimum allowed output range before falling back to unit scale.
    """

    def __init__(
        self,
        multi_fidelity: bool = False,
        num_fidelities: int = 1,
        fidelity_dim: int = -1,
        eps: float = 1e-12,
    ) -> None:
        self.multi_fidelity = multi_fidelity
        self.num_fidelities = num_fidelities
        self.fidelity_dim = fidelity_dim
        self.eps = eps
        self.y_min: Optional[torch.Tensor] = None
        self.y_max: Optional[torch.Tensor] = None
        self.y_range: Optional[torch.Tensor] = None

    def fit(self, Y: torch.Tensor, X: Optional[torch.Tensor] = None) -> "OutputScaler":
        """
        Fit scaling constants from output data.
        """
        Y = Y.float()
        if self.multi_fidelity:
            if X is None:
                raise ValueError("X is required to fit per-fidelity output scaling.")
            fidelity_indices = X[..., self.fidelity_dim].round().long()
            output_dim = Y.shape[-1]
            y_min = torch.zeros(self.num_fidelities, output_dim, dtype=Y.dtype, device=Y.device)
            y_max = torch.ones(self.num_fidelities, output_dim, dtype=Y.dtype, device=Y.device)
            for fidelity in range(self.num_fidelities):
                fidelity_mask = fidelity_indices == fidelity
                if fidelity_mask.any():
                    y_fidelity = Y[fidelity_mask]
                    y_min[fidelity] = y_fidelity.amin(dim=0).detach()
                    y_max[fidelity] = y_fidelity.amax(dim=0).detach()
            self.y_min = y_min
            self.y_max = y_max
        else:
            self.y_min = Y.amin(dim=-2, keepdim=True).detach()
            self.y_max = Y.amax(dim=-2, keepdim=True).detach()

        y_range = self.y_max - self.y_min
        self.y_range = torch.where(y_range.abs() >= self.eps, y_range, torch.ones_like(y_range))
        return self

    def to(self, device: torch.device) -> "OutputScaler":
        """
        Move scaling constants to a device.
        """
        if self.y_min is not None:
            self.y_min = self.y_min.to(device)
        if self.y_max is not None:
            self.y_max = self.y_max.to(device)
        if self.y_range is not None:
            self.y_range = self.y_range.to(device)
        return self

    def transform(self, Y: torch.Tensor, X: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Scale outputs to model-native units.
        """
        y_min, y_range = self._scaling_tensors(Y, X)
        return (Y - y_min) / y_range

    def inverse_transform(self, Y: torch.Tensor, X: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Transform scaled means or samples back to original output units.
        """
        y_min, y_range = self._scaling_tensors(Y, X)
        return Y * y_range + y_min

    def inverse_variance(self, Yvar: torch.Tensor, X: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Transform scaled variances back to original output units.
        """
        _, y_range = self._scaling_tensors(Yvar, X)
        return Yvar * y_range.pow(2)

    def inverse_covariance(self, covariance: torch.Tensor, X: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Transform a q x q covariance matrix back to original output units.
        """
        if self.multi_fidelity:
            if X is None:
                raise ValueError("X is required for per-fidelity covariance scaling.")
            fidelity_indices = X[..., self.fidelity_dim].round().long()
            scale = self.y_range[fidelity_indices].squeeze(-1)
            return covariance * scale.unsqueeze(-1) * scale.unsqueeze(-2)
        scale = self.y_range.squeeze()
        return covariance * scale.pow(2)

    def state_dict(self) -> Dict[str, Any]:
        """
        Return serializable scaler state.
        """
        return {
            "multi_fidelity": self.multi_fidelity,
            "num_fidelities": self.num_fidelities,
            "fidelity_dim": self.fidelity_dim,
            "eps": self.eps,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "y_range": self.y_range,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any], device: Optional[torch.device] = None) -> "OutputScaler":
        """
        Reconstruct a scaler from serialized state.
        """
        scaler = cls(
            multi_fidelity=state.get("multi_fidelity", False),
            num_fidelities=state.get("num_fidelities", 1),
            fidelity_dim=state.get("fidelity_dim", -1),
            eps=state.get("eps", 1e-12),
        )
        scaler.y_min = state["y_min"]
        scaler.y_max = state["y_max"]
        if "y_range" in state:
            scaler.y_range = state["y_range"]
        else:
            y_range = scaler.y_max - scaler.y_min
            scaler.y_range = torch.where(y_range.abs() >= scaler.eps, y_range, torch.ones_like(y_range))
        if device is not None:
            scaler.to(device)
        return scaler

    def _scaling_tensors(self, Y: torch.Tensor, X: Optional[torch.Tensor]) -> tuple:
        if self.y_min is None or self.y_range is None:
            raise RuntimeError("OutputScaler must be fit before use.")
        if self.multi_fidelity:
            if X is None:
                raise ValueError("X is required for per-fidelity output scaling.")
            fidelity_indices = X[..., self.fidelity_dim].round().long().to(self.y_min.device)
            y_min = self.y_min[fidelity_indices]
            y_range = self.y_range[fidelity_indices]
        else:
            y_min = self.y_min
            y_range = self.y_range
        return y_min.to(Y.device), y_range.to(Y.device)
