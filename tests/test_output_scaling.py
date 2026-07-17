import torch

from deepopt.output_scaling import OutputScaler, StandardizeOutputScaler


def test_output_scaler_single_fidelity_round_trip():
    X = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float32)
    Y = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    scaler = OutputScaler().fit(Y, X)

    Y_scaled = scaler.transform(Y, X)

    torch.testing.assert_close(Y_scaled, torch.tensor([[0.0], [0.5], [1.0]]))
    torch.testing.assert_close(scaler.inverse_transform(Y_scaled, X), Y)
    torch.testing.assert_close(scaler.inverse_variance(torch.ones_like(Y), X), 4.0 * torch.ones_like(Y))


def test_output_scaler_single_fidelity_handles_batched_outputs():
    Y = torch.tensor([[[1.0], [3.0]], [[10.0], [20.0]]], dtype=torch.float32)
    scaler = OutputScaler().fit(Y)

    Y_scaled = scaler.transform(Y)

    torch.testing.assert_close(Y_scaled, torch.tensor([[[0.0], [1.0]], [[0.0], [1.0]]]))
    torch.testing.assert_close(scaler.inverse_transform(Y_scaled), Y)


def test_output_scaler_single_fidelity_inverse_covariance_handles_scalar_scale():
    Y = torch.tensor([[1.0], [3.0]], dtype=torch.float32)
    scaler = OutputScaler().fit(Y)
    cov = torch.ones(2, 2, dtype=torch.float32)

    cov_original = scaler.inverse_covariance(cov)

    torch.testing.assert_close(cov_original, 4.0 * torch.ones(2, 2))


def test_output_scaler_multi_fidelity_uses_independent_ranges():
    X = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    Y = torch.tensor([[10.0], [20.0], [100.0], [300.0]], dtype=torch.float32)
    scaler = OutputScaler(multi_fidelity=True, num_fidelities=2, fidelity_dim=1).fit(Y, X)

    Y_scaled = scaler.transform(Y, X)

    torch.testing.assert_close(Y_scaled, torch.tensor([[0.0], [1.0], [0.0], [1.0]]))
    torch.testing.assert_close(scaler.inverse_transform(Y_scaled, X), Y)
    torch.testing.assert_close(
        scaler.inverse_variance(torch.ones_like(Y), X),
        torch.tensor([[100.0], [100.0], [40000.0], [40000.0]]),
    )


def test_output_scaler_multi_fidelity_inverse_covariance_handles_mixed_q():
    X_train = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    Y_train = torch.tensor([[0.0], [2.0], [10.0], [20.0]], dtype=torch.float32)
    scaler = OutputScaler(multi_fidelity=True, num_fidelities=2, fidelity_dim=1).fit(Y_train, X_train)
    X_query = torch.tensor([[0.1, 0.0], [0.9, 1.0]], dtype=torch.float32)
    cov = torch.ones(2, 2, dtype=torch.float32)

    cov_original = scaler.inverse_covariance(cov, X_query)

    torch.testing.assert_close(cov_original, torch.tensor([[4.0, 20.0], [20.0, 100.0]]))


def test_standardize_output_scaler_round_trip():
    scaler = StandardizeOutputScaler(mean=torch.tensor([[10.0]]), std=torch.tensor([[2.0]]))
    Y = torch.tensor([[8.0], [10.0], [14.0]], dtype=torch.float32)

    Y_scaled = scaler.transform(Y)

    torch.testing.assert_close(Y_scaled, torch.tensor([[-1.0], [0.0], [2.0]]))
    torch.testing.assert_close(scaler.inverse_transform(Y_scaled), Y)
    torch.testing.assert_close(scaler.inverse_variance(torch.ones_like(Y)), 4.0 * torch.ones_like(Y))


def test_standardize_output_scaler_inverse_covariance():
    scaler = StandardizeOutputScaler(mean=torch.tensor([[10.0]]), std=torch.tensor([[2.0]]))
    cov = torch.ones(2, 2, dtype=torch.float32)

    cov_original = scaler.inverse_covariance(cov)

    torch.testing.assert_close(cov_original, 4.0 * torch.ones(2, 2))


def test_standardize_output_scaler_from_legacy_state_dict():
    scaler = StandardizeOutputScaler.from_botorch_state_dict(
        {
            "outcome_transform.means": torch.tensor([[10.0]]),
            "outcome_transform.stdvs": torch.tensor([[2.0]]),
        }
    )

    torch.testing.assert_close(scaler.transform(torch.tensor([[14.0]])), torch.tensor([[2.0]]))


def test_standardize_output_scaler_from_legacy_stdvs_sq():
    scaler = StandardizeOutputScaler.from_botorch_state_dict(
        {
            "outcome_transform.means": torch.tensor([[10.0]]),
            "outcome_transform._stdvs_sq": torch.tensor([[4.0]]),
        }
    )

    torch.testing.assert_close(scaler.transform(torch.tensor([[14.0]])), torch.tensor([[2.0]]))
