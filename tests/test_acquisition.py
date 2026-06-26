import inspect

import pytest

pytest.importorskip("botorch")
pytest.importorskip("gpytorch")

from deepopt import acquisition

pytestmark = pytest.mark.requires_botorch


def test_acquisition_public_classes_are_importable():
    assert acquisition.qMaxValueEntropy.__name__ == "qMaxValueEntropy"
    assert acquisition.qMultiFidelityMaxValueEntropy.__name__ == "qMultiFidelityMaxValueEntropy"
    assert acquisition.qMultiFidelityLowerBoundMaxValueEntropy.__name__ == "qMultiFidelityLowerBoundMaxValueEntropy"


def test_qmaxvalueentropy_constructor_signature_regression():
    sig = inspect.signature(acquisition.qMaxValueEntropy)

    expected = [
        "model",
        "candidate_set",
        "num_fantasies",
        "num_mv_samples",
        "num_y_samples",
        "posterior_transform",
        "use_gumbel",
        "maximize",
        "X_pending",
        "train_inputs",
    ]
    assert list(sig.parameters)[: len(expected)] == expected
    assert sig.parameters["num_fantasies"].default == 16
    assert sig.parameters["num_mv_samples"].default == 10
    assert sig.parameters["num_y_samples"].default == 128
    assert sig.parameters["use_gumbel"].default is True
    assert sig.parameters["maximize"].default is True


def test_multifidelity_constructor_signature_regression():
    sig = inspect.signature(acquisition.qMultiFidelityMaxValueEntropy)

    for name in ["cost_aware_utility", "project", "expand"]:
        assert name in sig.parameters
    assert sig.parameters["num_fantasies"].default == 16
    assert sig.parameters["num_mv_samples"].default == 10
    assert sig.parameters["num_y_samples"].default == 128
