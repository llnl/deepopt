"""
This module contains the neural network modules used throughout
DeepOpt. This includes MLP and SIREN neural networks.
"""
from math import cos, pi
from typing import Type, Union

import numpy as np
import torch
from torch import nn
from torch.optim import SGD, Adam

from deepopt.configuration import ConfigSettings


class MLPLayer(nn.Module):
    """
    A class representation for a layer of an MLP neural network.
    """

    def __init__(self, config: ConfigSettings, input_dim: int, output_dim: int, is_first: bool, is_last: bool):
        """
        Create a layer of the MLP neural network.

        :param config: Configuration settings provided by the user
        :param input_dim: The size of the input sample
        :param output_dim: The size of the output sample
        :param is_first: If True, this is the first layer in our MLP network.
            Weights in a SIREN network are initialized differently in first layer.
        :param is_last: If True, this is the last layer in our MLP network so don't apply any dropout,
            batch normalization, or activation.
        """
        super().__init__()

        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_first = is_first
        self.is_last = is_last

        self.linear = nn.Linear(input_dim, output_dim)

        # Grab the weight (only used for SIREN)
        self.w0 = self.config.get_setting("w0")

        # Set the activation function
        activation_mapper = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "identity": nn.Identity(),
            "siren": lambda x: torch.sin(self.w0 * x),
        }
        self.activation = self.config.get_setting("activation")
        try:
            self.activation_fn = activation_mapper[self.activation]
        except KeyError as exc:
            raise NotImplementedError(f"The only activations that are supported are {activation_mapper.keys()}") from exc
        if self.activation == "siren":
            self.init_weights()

    def init_weights(self):
        """
        Initialize the weights for this layer
        """
        b = 1 / self.input_dim if self.is_first else np.sqrt(6 / self.input_dim) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass computation for this layer.

        :param x: The input tensor for this layer

        :returns: The output tensor for this layer
        """
        # Apply linear transformation
        x = self.linear(x)

        # If this is the last layer, don't apply activation, batchnorm, or dropout
        if self.is_last:
            return self.w0 * x if self.activation == "siren" else x

        # Grab the activation and dropout settings
        activation_first = self.config.get_setting("activation_first")
        dropout_setting = self.config.get_setting("dropout")
        dropout_fn = nn.Dropout(self.config.get_setting("dropout_prob"))

        # Apply the dropout first if we're not doing activation first
        if dropout_setting and not activation_first:
            x = dropout_fn(x)

        # Apply the activation function
        x = self.activation_fn(x)

        # Apply batchnorm if necessary
        if self.config.get_setting("batchnorm"):
            x = nn.BatchNorm1d(self.output_dim)(x)

        # Apply the dropout last if we're doing activation first
        if dropout_setting and activation_first:
            x = dropout_fn(x)

        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) neural network module.
    This uses a nonlinear activation function to train a model.
    """

    def __init__(
        self,
        config: ConfigSettings,
        unc_type: str,
        input_dim: int,
        output_dim: int,
        device: str = "cpu",
    ):
        """
        Create an MLP neural network

        :param config: Configuration settings provided by the user
        :param unc_type: Type of encoding. Options: "deltaenc" or anything else
        :param input_dim: The size of the input sample
        :param output_dim: The size of the output sample
        :param device: Which device to run the neural network on.
            Options: 'cpu' or 'gpu'.
        """
        super().__init__()
        self.config = config
        self.unc_type = unc_type
        self.device = device

        if self.config.get_setting("ff"):
            scale = np.sqrt(self.config.get_setting("variance"))  # /(input_dim-1)
            dist = self.config.get_setting("dist")
            mapping_size = self.config.get_setting("mapping_size")
            if dist == "uniform":
                mn = -scale
                mx = scale
                self.B = torch.rand((mapping_size, input_dim)) * (mx - mn) + mn
            elif dist == "gaussian":
                self.B = torch.randn((mapping_size, input_dim)) * scale
            elif dist == "laplace":
                rp = np.random.laplace(loc=0.0, scale=scale, size=(mapping_size, input_dim))
                self.B = torch.from_numpy(rp).float()
            self.B = self.B.to(device)
            if self.unc_type == "deltaenc":
                first_layer_dim = mapping_size * 4
            else:
                first_layer_dim = mapping_size * 2
        else:
            self.B = None
            if self.unc_type == "deltaenc":
                first_layer_dim = 2 * input_dim
            else:
                first_layer_dim = input_dim

        layers = []
        n_layers = self.config.get_setting("n_layers")
        hidden_dim = self.config.get_setting("hidden_dim")
        for i in range(n_layers):
            is_first = i == 0
            is_last = i == (n_layers - 1)
            input_dim_lyr = first_layer_dim if is_first else hidden_dim
            output_dim_lyr = output_dim if is_last else hidden_dim
            layers.append(
                MLPLayer(
                    config=self.config,
                    input_dim=input_dim_lyr,
                    output_dim=output_dim_lyr,
                    is_first=is_first,
                    is_last=is_last,
                )
            )

        self.mlp = nn.Sequential(*layers).to(device)

    def input_mapping(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform the input data into a format that can be processed by the MLP
        neural network.

        :param x: The tensor of input data to transform

        :returns: The tensor of transformed input data
        """
        if self.B is None:
            return x.to(self.device)

        x_proj = (2.0 * np.pi * x).float().to(self.device) @ self.B.t()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass computation for the MLP neural network

        :param x: The input tensor to the neural network

        :returns: The output tensor computed from the forward pass
        """
        if self.unc_type == "deltaenc":
            out = self.mlp(x.to(self.device))
        else:
            h = self.input_mapping(x.to(self.device))
            out = self.mlp(h)
        return out


def create_optimizer(network: Type[nn.Module], config: ConfigSettings) -> Union[Adam, SGD]:
    """
    This function instantiates and returns optimizer objects of the input neural network

    :param network: The input neural network
    :param config: The configuration options provided by the user
    """
    opt_type = config.get_setting("opt_type")
    weight_decay = config.get_setting("weight_decay")
    if opt_type == "Adam":
        optimizer = Adam(
            network.parameters(),
            lr=config.get_setting("learning_rate"),
            weight_decay=weight_decay,
        )

    elif opt_type == "SGD":
        optimizer = SGD(
            network.parameters(),
            lr=config.get_setting("learning_rate"),
            weight_decay=weight_decay,
        )

    else:
        raise NotImplementedError("Only Adam and SGD optimizers supported as of now")

    return optimizer


def proposed_lr(config, epoch, epoch_per_cycle):
    """
    Compute a cosine-annealed learning rate for one snapshot ensemble cycle.

    :param config: Mapping containing ``learning_rate``.
    :param epoch: Current training epoch.
    :param epoch_per_cycle: Number of epochs in one cosine cycle.
    :returns: The learning rate for ``epoch``.
    """
    # Cosine Annealing Learning Rate Update
    # https://github.com/moskomule/pytorch.snapshot.ensembles/blob/master/se.py
    iteration = int(epoch % epoch_per_cycle)
    return config["learning_rate"] * (cos(pi * iteration / epoch_per_cycle) + 1) / 2


def prepare_cut_mix_batch(config, input, target):
    """
    Create a CutMix-style mixed batch for tabular inputs.

    :param config: Mapping containing the beta-distribution parameter ``beta``.
    :param input: Input batch with shape ``n x d``.
    :param target: Target batch aligned with ``input``.
    :returns: ``(mixed_input, target_a, target_b, lam)`` for CutMix loss evaluation.
    """
    # Generate Mixed Sample
    # https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py
    lam = np.random.beta(config["beta"], config["beta"])
    rand_index = torch.randperm(input.size()[0])
    target_a = target
    target_b = target[rand_index]

    num_dim_mixed = np.random.randint(input.size()[1] // 2)
    mix_dim = torch.LongTensor(np.random.choice(range(input.size()[1]), num_dim_mixed))

    input[:, mix_dim] = input[(rand_index), :][:, (mix_dim)]
    return input, target_a, target_b, lam
