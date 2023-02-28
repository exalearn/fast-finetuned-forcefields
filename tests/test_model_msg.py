"""Tests dealing with the classes I use to transmit models between workers"""
import pickle as pkl

from pytest import fixture
import numpy as np
import torch

from fff.learning.util.messages import TorchMessage


@fixture()
def model() -> torch.nn.Module:
    return torch.nn.Linear(in_features=4, out_features=1)


def test_pickle(model):
    """Make sure we can pickle then get the model back"""

    # Create the message
    model_msg = TorchMessage(model)
    assert model_msg._pickle is None

    # Make a copy
    model_msg_copy: TorchMessage = pkl.loads(pkl.dumps(model_msg))
    assert model_msg._pickle is None  # Original should not hold on to the serialized version
    assert model_msg_copy._pickle is not None  # The new one should not have a model yet
    assert model_msg_copy.model is None

    # Ensure that the new model yields the same result as the original
    model_copy = model_msg_copy.get_model()
    x = torch.Tensor([0, 1, 2, 4])
    assert np.isclose(model(x).detach(), model_copy(x).detach()).all()
