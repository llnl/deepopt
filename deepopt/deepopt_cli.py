"""
This module establishes the entrypoint to the DeepOpt library and handles the
creation of all commands and options via the
[Click](https://click.palletsprojects.com/en/latest/) library.
"""
import json
from gettext import ngettext
from typing import Any, List, Mapping, Tuple, Union

import click
import torch
import numpy as np
from click.core import iter_params_for_processing

from deepopt.configuration import ConfigSettings
from deepopt.defaults import Defaults
from deepopt.models import DelUQModel, GPModel, NNEnsembleModel, get_checkpoint_metadata, load_deepopt_wrapper


def get_deepopt_model(model_type: str) -> Union[GPModel, DelUQModel, NNEnsembleModel]:
    """
    Given the type of model by the user, return the correct model
    object from the DeepOpt library to use for training/optimizing.

    :param model_type: The type of surrogate model to use

    :returns: A DeepOpt model to use for training/optimizing
    """
    if model_type == "GP":
        deepopt_model = GPModel
    elif model_type == "delUQ":
        deepopt_model = DelUQModel
    elif model_type == "nnEnsemble":
        deepopt_model = NNEnsembleModel
    else:
        raise ValueError(f"The model type {model_type} is not a valid DeepOpt model. Valid models are 'GP', 'delUQ', and 'nnEnsemble'.")

    return deepopt_model

class DeepoptCommand(click.Command):
    """
    A custom click command to help accommodate the `ConditionalOption`
    functionality we created.
    """

    def parse_args(self, ctx: click.Context, args: List[str]) -> List[str]:
        """
        Override Click's default arg parse functionality with one that
        accommodates the `ConditionalOption` we introduce. Most of this code
        is taken directly from
        [Click's API on GitHub](https://github.com/pallets/click/blob/main/src/click/core.py#L1149).

        :param ctx: A click context object
        :param args: A list of args passed in by the user

        :returns: A list of parsed args
        """
        if not args and self.no_args_is_help and not ctx.resilient_parsing:
            click.echo(ctx.get_help(), color=ctx.color)
            ctx.exit()

        parser = self.make_parser(ctx)
        opts, args, param_order = parser.parse_args(args=args)

        # Custom functionality to place conditional options last in the param order
        conditional_opts = []
        nonconditional_opts = []
        for param in param_order:
            if isinstance(param, ConditionalOption):
                conditional_opts.append(param)
            else:
                nonconditional_opts.append(param)
        param_order = nonconditional_opts + conditional_opts

        for param in iter_params_for_processing(param_order, self.get_params(ctx)):
            value, args = param.handle_parse_result(ctx, opts, args)

        if args and not ctx.allow_extra_args and not ctx.resilient_parsing:
            ctx.fail(
                ngettext(
                    "Got unexpected extra argument ({args})",
                    "Got unexpected extra arguments ({args})",
                    len(args),
                ).format(args=" ".join(map(str, args)))
            )

        ctx.args = args
        ctx._opt_prefixes.update(parser._opt_prefixes)
        return args


class ConditionalOption(click.Option):
    """
    A custom click option to represent a conditional option.
    Conditional options are click options that depend on independent option(s)
    and therefore cannot be used without said independent option(s).
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the conditional option by saving which independent option(s)
        this depends on. In some cases we may also need to save what the independent
        option(s) must be equal to so we'll grab that value too if necessary.
        """
        self.depends_on = kwargs.pop("depends_on")
        self.equal_to = kwargs.pop("equal_to", None)
        kwargs["help"] = f"[NOTE: This argument is only used if {self.depends_on} is used.] " + kwargs.get("help", "")
        super().__init__(*args, **kwargs)

    def handle_parse_result(self, ctx: click.Context, opts: Mapping[str, Any], args: List[str]) -> Tuple[Any, List[str]]:
        """
        After click parses the result, validate the result and check if our conditions
        are satisfied to use this option.

        :param ctx: A click context object storing the params passed in by the user
        :param opts: A mapping of options provided for this option
        :param args: A list of args provided for this option

        :returns: A tuple containing the value of this option that was given by the
            user and the args that the user provided
        """
        value, args = super().handle_parse_result(ctx, opts, args)
        is_conditional_opt_used = self.name in opts
        is_dependency_used = bool(ctx.params.get(self.depends_on, None))
        if is_dependency_used and self.equal_to is not None:
            is_dependency_used = ctx.params.get(self.depends_on) == self.equal_to

        if is_conditional_opt_used and not is_dependency_used:
            equal_to_str = self.equal_to if self.equal_to is not None else "not None"
            click.echo(
                f"Option {self.name} will not be used and is set to None. "
                f"Used only when {self.depends_on} is {equal_to_str}"
            )
            value = None
            ctx.params[self.name] = value
        return value, args


@click.group()
def deepopt_cli():
    """
    A simple and easy-to-use library for performing Bayesian optimization.
    """


@deepopt_cli.command(cls=DeepoptCommand)
@click.option(
    "-i",
    "--infile",
    help="Input data to train from.",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "-o",
    "--outfile",
    help="Outfile to save model checkpoint.",
    type=click.STRING,
    required=True,
)
@click.option(
    "-b",
    "--bounds",
    help="Bounds for each input dimension.",
    type=click.STRING,
    required=True,
)
@click.option(
    "-m",
    "--model-type",
    help="What kind of surrogate are you using?",
    default=Defaults.model_type,
    show_default=True,
    type=click.Choice(["GP", "delUQ","nnEnsemble"]),
)
@click.option(
    "-c",
    "--config-file",
    help="Config file containing hyper parameters.",
    type=click.Path(exists=True),
    default=None,
)
@click.option(
    "-r",
    "--random-seed",
    help="Random seed.",
    default=Defaults.random_seed,
    show_default=True,
    type=click.INT,
)
@click.option(
    "-k",
    "--k-folds",
    help="Number of k-folds.",
    default=Defaults.k_folds,
    show_default=True,
    type=click.INT,
)
@click.option(
    "-d",
    "--device",
    help="Device to use (cpu/gpu/auto)",
    default="auto",
    show_default=True,
    type=click.Choice(["cpu","gpu","cuda","auto"])
)
@click.option(
    "--multi-fidelity",
    help="Single or multi-fidelity?",
    is_flag=True,
    default=Defaults.multi_fidelity,
    type=click.BOOL,
    show_default=True,
)
def learn(
    infile,
    outfile,
    bounds,
    model_type,
    config_file,
    random_seed,
    k_folds,
    device,
    multi_fidelity,
) -> None:
    """
    Train a model on a dataset and save that model to an output file.
    """
    bounds = np.array(json.loads(bounds),dtype=np.float32).T

    config_settings = ConfigSettings(model_type, config_file=config_file)

    deepopt_model = get_deepopt_model(model_type)

    model = deepopt_model(
        config_settings=config_settings,
        data_file=infile,
        multi_fidelity=multi_fidelity,
        random_seed=random_seed,
        bounds=bounds,
        k_folds=k_folds,
        device=device,
    )
    model.learn(outfile=outfile)


@deepopt_cli.command(cls=DeepoptCommand)
@click.option(
    "-i",
    "--infile",
    help="Training data path.",
    type=click.Path(exists=True),
    required=False,
)
@click.option(
    "-o",
    "--outfile",
    help="Where to place the suggested candidates.",
    type=click.STRING,
    required=True,
)
@click.option(
    "-l",
    "--learner-file",
    help="Learner path. Ex: /learners/my_learner.ckpt",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "-b",
    "--bounds",
    help="Bounds for each input dimension.",
    type=click.STRING,
    required=False,
)
@click.option(
    "-a",
    "--acq-method",
    help="""
              \b
              The acquisition function.
              [NOTE: Some acquisition functions only work with a specific fidelity.]
              \b
              Single    - [KG|MaxValEntropy|EI|NEI]
              Multi     - [KG|MaxValEntropy]
              """,
    type=click.Choice(["EI", "NEI", "KG", "MaxValEntropy"]),
    required=True,
)
@click.option(
    "-m",
    "--model-type",
    help="What kind of surrogate are you using?",
    show_default=True,
    default=Defaults.model_type,
    type=click.Choice(["GP", "delUQ","nnEnsemble"]),
)
@click.option(
    "-c",
    "--config-file",
    help="Config file containing hyper parameters.",
    type=click.Path(exists=True),
    default=None,
)
@click.option(
    "-r",
    "--random-seed",
    help="Random seed.",
    default=Defaults.random_seed,
    show_default=True,
    type=click.INT,
)
@click.option(
    "-q",
    "--num-candidates",
    help="The number of candidates.",
    default=Defaults.num_candidates,
    type=click.INT,
    show_default=True,
)
@click.option(
    "-d",
    "--device",
    help="Device to use (cpu/gpu/auto)",
    default="auto",
    show_default=True,
    type=click.Choice(["cpu","gpu","cuda","auto"])
)
@click.option(
    "--multi-fidelity",
    help="Single or multi-fidelity?",
    is_flag=True,
    default=Defaults.multi_fidelity,
    show_default=True,
    type=click.BOOL,
)
@click.option(
    "--fidelity-cost",
    help="List of costs for each fidelity.",
    type=click.STRING,
    default=Defaults.fidelity_cost,
    show_default=True,
)
@click.option(
    "-v",
    "--verbose",
    help="Print details of model evaluations and fantasy training.",
    is_flag=True,
    default=False,
    show_default=True,
    type=click.BOOL
)
@click.option(
    "--risk-measure",
    help="The risk measure to apply.",
    type=click.Choice(["VaR", "CVaR"]),
)
@click.option(
    "--risk-level",
    help="The risk level.",
    type=click.FloatRange(0, 1, min_open=True),
    cls=ConditionalOption,
    depends_on="risk_measure",
)
@click.option(
    "--risk-n-deltas",
    help="The number of input perturbations to sample for X's uncertainty. [example: --risk-n-deltas 10].",
    type=click.INT,
    cls=ConditionalOption,
    depends_on="risk_measure",
)
@click.option(
    "--X-stddev",
    help="Uncertainty in X (stddev) in each dimension. [example: --X-stddev [0.00005]].",
    type=click.STRING,
    cls=ConditionalOption,
    depends_on="risk_measure",
)
@click.option(
    "--propose-best",
    help="Select first candidate using surrogate optimum.",
    is_flag=True,
    type=click.BOOL,
    default=False,
    show_default=True,
)
@click.option(
    "--integer-fidelities",
    help="Convert fidelity column to integers when saving.",
    is_flag=True,
    type=click.BOOL,
    default=False,
    show_default=True,
)
def optimize(
    infile,
    outfile,
    config_file,
    learner_file,
    bounds,
    acq_method,
    random_seed,
    model_type,
    num_candidates,
    device,
    multi_fidelity,
    fidelity_cost,
    verbose,
    risk_measure,
    risk_level,
    risk_n_deltas,
    x_stddev,
    propose_best,
    integer_fidelities,
) -> None:
    """
    Load in the model created by ``learn`` and use it to propose new simulation points.

    Self-describing checkpoints provide their own training data, bounds, model type,
    and config settings. Legacy checkpoints still require JSON-encoded ``--bounds``
    and the original ``--infile``.
    """
    checkpoint_metadata = get_checkpoint_metadata(learner_file)
    if checkpoint_metadata is None:
        # Legacy checkpoints do not include the data and bounds needed to rebuild a wrapper.
        if infile is None:
            raise click.UsageError("Legacy checkpoints require --infile.")
        if bounds is None:
            raise click.UsageError("Legacy checkpoints require --bounds.")
        bounds = np.array(json.loads(bounds),dtype=np.float32).T
        config_settings = ConfigSettings(model_type, config_file=config_file)
        deepopt_model = get_deepopt_model(model_type)
        model = deepopt_model(
            config_settings=config_settings,
            data_file=infile,
            multi_fidelity=multi_fidelity,
            random_seed=random_seed,
            bounds=bounds,
            device=device,
            verbose=verbose,
        )
    else:
        model = load_deepopt_wrapper(learner_file, device=device, verbose=verbose)
        if config_file is not None:
            optimize_config_settings = ConfigSettings(checkpoint_metadata["model_type"], config_file=config_file)
            if "optimization" in optimize_config_settings:
                model.config_settings.set_setting(
                    "optimization",
                    optimize_config_settings.get_setting("optimization"),
                )

    risk_measure = None if risk_measure == "None" else risk_measure
    if risk_measure:
        # x_stddev = torch.tensor(json.loads(x_stddev),dtype=torch.float,device=device)
        x_stddev = np.array(json.loads(x_stddev),dtype=np.float32)
    if model.multi_fidelity:
        # fidelity_cost = torch.tensor(json.loads(fidelity_cost),dtype=torch.float,device=device)
        fidelity_cost = np.array(json.loads(fidelity_cost),dtype=np.float32)
    else:
        fidelity_cost = None
        integer_fidelities = False
    model.optimize(
        outfile=outfile,
        learner_file=learner_file,
        acq_method=acq_method,
        num_candidates=num_candidates,
        fidelity_cost=fidelity_cost,
        risk_measure=risk_measure,
        risk_level=risk_level,
        risk_n_deltas=risk_n_deltas,
        x_stddev=x_stddev,
        propose_best=propose_best,
        integer_fidelities=integer_fidelities,
    )


def main():
    """The entrypoint to the DeepOpt library."""
    deepopt_cli(max_content_width=800)


if __name__ == "__main__":
    main()
