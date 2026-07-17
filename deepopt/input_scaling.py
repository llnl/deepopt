"""
Shared input scaling utilities for DeepOpt models.
"""
from typing import Any, Dict, Optional

import torch


DEPRECATED_ORIGINAL_SCALE_ERROR = (
    "original_scale was renamed; use original_scale_x for input units and "
    "original_scale_y for output units."
)


def reject_deprecated_original_scale(args: tuple, kwargs: Dict[str, Any]) -> None:
    """
    Reject the deprecated ``original_scale`` prediction argument.

    :param args: Positional arguments passed after ``get_cov``.
    :param kwargs: Keyword arguments passed to a prediction method.
    :raises TypeError: If callers use the removed ``original_scale`` argument.
    """
    if args or "original_scale" in kwargs:
        raise TypeError(DEPRECATED_ORIGINAL_SCALE_ERROR)


class InputScaler:
    """
    Min/max input scaler with optional unscaled fidelity indices.

    :param bounds: Tensor-like array with shape ``2 x input_dim`` in original input units.
    :param multi_fidelity: If ``True``, keep fidelity indices discrete during scaling.
    :param fidelity_dim: Dimension containing fidelity indices; defaults to the last column.
    :param eps: Minimum allowed range before falling back to unit scale.
    """

    def __init__(
        self,
        bounds: torch.Tensor,
        multi_fidelity: bool = False,
        fidelity_dim: int = -1,
        eps: float = 1e-12,
    ) -> None:
        bounds = torch.as_tensor(bounds, dtype=torch.float).detach()
        if bounds.ndim != 2 or bounds.shape[0] != 2:
            raise ValueError("bounds must have shape 2 x input_dim.")
        self.bounds = bounds
        self.multi_fidelity = multi_fidelity
        self.fidelity_dim = fidelity_dim
        self.eps = eps
        self.x_min = bounds[0]
        self.x_max = bounds[1]
        x_range = self.x_max - self.x_min
        self.x_range = torch.where(x_range.abs() >= self.eps, x_range, torch.ones_like(x_range))

    def to(self, device: torch.device) -> "InputScaler":
        """
        Move scaler tensors to a device.

        :param device: Target PyTorch device.
        :returns: This scaler, after moving stored tensors.
        """
        self.bounds = self.bounds.to(device)
        self.x_min = self.x_min.to(device)
        self.x_max = self.x_max.to(device)
        self.x_range = self.x_range.to(device)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Scale inputs from original units to model units.

        :param X: Tensor-like inputs whose last dimension matches ``input_dim``.
        :returns: Scaled inputs with the same shape as ``X``; fidelity indices are rounded when ``multi_fidelity=True``.
        :raises ValueError: If the last dimension of ``X`` does not match the scaler bounds.
        """
        X = torch.as_tensor(X).float()
        self._validate_input_dim(X)
        x_min, x_range = self._scaling_tensors(X)
        X_scaled = (X - x_min) / x_range
        if self.multi_fidelity:
            fidelity_dim = self._normalized_fidelity_dim(X)
            X_scaled[..., fidelity_dim] = X[..., fidelity_dim].round()
        return X_scaled

    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transform model-unit inputs back to original units.

        :param X: Tensor-like inputs whose last dimension matches ``input_dim``.
        :returns: Inputs in original units with the same shape as ``X``; fidelity indices are rounded when ``multi_fidelity=True``.
        :raises ValueError: If the last dimension of ``X`` does not match the scaler bounds.
        """
        X = torch.as_tensor(X).float()
        self._validate_input_dim(X)
        x_min, x_range = self._scaling_tensors(X)
        X_original = X * x_range + x_min
        if self.multi_fidelity:
            fidelity_dim = self._normalized_fidelity_dim(X)
            X_original[..., fidelity_dim] = X[..., fidelity_dim].round()
        return X_original

    def state_dict(self) -> Dict[str, Any]:
        """
        Return serializable input-scaler state for checkpoints.

        :returns: Dictionary containing bounds, fidelity settings, and cached scaling tensors.
        """
        return {
            "bounds": self.bounds,
            "multi_fidelity": self.multi_fidelity,
            "fidelity_dim": self.fidelity_dim,
            "eps": self.eps,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "x_range": self.x_range,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any], device: Optional[torch.device] = None) -> "InputScaler":
        """
        Reconstruct an input scaler from checkpoint state.

        :param state: State produced by ``state_dict``.
        :param device: Optional target device for restored tensors.
        :returns: A reconstructed ``InputScaler``.
        """
        scaler = cls(
            bounds=state["bounds"],
            multi_fidelity=state.get("multi_fidelity", False),
            fidelity_dim=state.get("fidelity_dim", -1),
            eps=state.get("eps", 1e-12),
        )
        scaler.x_min = state.get("x_min", scaler.bounds[0])
        scaler.x_max = state.get("x_max", scaler.bounds[1])
        if "x_range" in state:
            scaler.x_range = state["x_range"]
        else:
            x_range = scaler.x_max - scaler.x_min
            scaler.x_range = torch.where(x_range.abs() >= scaler.eps, x_range, torch.ones_like(x_range))
        if device is not None:
            scaler.to(device)
        return scaler

    def _validate_input_dim(self, X: torch.Tensor) -> None:
        if X.shape[-1] != self.bounds.shape[-1]:
            raise ValueError(
                f"Expected input last dimension {self.bounds.shape[-1]}, found tensor of shape {X.shape}."
            )

    def _normalized_fidelity_dim(self, X: torch.Tensor) -> int:
        return self.fidelity_dim if self.fidelity_dim >= 0 else X.shape[-1] + self.fidelity_dim

    def _scaling_tensors(self, X: torch.Tensor) -> tuple:
        return self.x_min.to(X.device), self.x_range.to(X.device)
