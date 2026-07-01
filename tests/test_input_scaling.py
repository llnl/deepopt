import pytest
import torch

from deepopt.input_scaling import InputScaler


def test_single_fidelity_transform_and_inverse_non_unit_bounds():
    bounds = torch.tensor([[10.0, -2.0], [20.0, 2.0]])
    scaler = InputScaler(bounds)
    X = torch.tensor([[10.0, -2.0], [15.0, 0.0], [20.0, 2.0]])

    X_scaled = scaler.transform(X)

    torch.testing.assert_close(X_scaled, torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]))
    torch.testing.assert_close(scaler.inverse_transform(X_scaled), X)


def test_multi_fidelity_transform_leaves_fidelity_unscaled_and_rounded():
    bounds = torch.tensor([[0.0, 10.0, 0.0], [10.0, 20.0, 3.0]])
    scaler = InputScaler(bounds, multi_fidelity=True, fidelity_dim=-1)
    X = torch.tensor([[5.0, 15.0, 1.2], [10.0, 10.0, 2.8]])

    X_scaled = scaler.transform(X)

    torch.testing.assert_close(X_scaled, torch.tensor([[0.5, 0.5, 1.0], [1.0, 0.0, 3.0]]))
    torch.testing.assert_close(scaler.inverse_transform(X_scaled), torch.tensor([[5.0, 15.0, 1.0], [10.0, 10.0, 3.0]]))


def test_state_dict_round_trip_and_to_cpu():
    bounds = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
    scaler = InputScaler(bounds, eps=1e-9)

    loaded = InputScaler.from_state_dict(scaler.state_dict(), device=torch.device("cpu"))

    torch.testing.assert_close(loaded.bounds, bounds)
    torch.testing.assert_close(loaded.transform(torch.tensor([[2.0, 4.0]])), torch.tensor([[0.5, 0.5]]))
    assert loaded.eps == 1e-9


def test_transform_rejects_wrong_input_dimension():
    scaler = InputScaler(torch.tensor([[0.0, 0.0], [1.0, 1.0]]))

    with pytest.raises(ValueError, match="Expected input last dimension"):
        scaler.transform(torch.tensor([[0.0, 0.0, 0.0]]))
