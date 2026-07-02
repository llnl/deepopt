"""
This module contains the class objects that will be used
to load and store configuration settings.
"""
import json
from typing import Any

import yaml

from deepopt.defaults import DELUQ_CONFIG, GP_CONFIG, NNENSEMBLE_CONFIG


class ConfigSettings:
    """
    Class for loading and storing configuration settings.
    """

    def __init__(self, model_type: str, config_file: str = None):
        """
        Set up the configurations by loading them from `config_file`.

        :param config_file: A path to the configuration file to load
        :param model_type: A str representing the type of model (`GP`, `delUQ`, or `nnEnsemble`)
        """
        self.config_file = config_file
        self.model_type = model_type

        if self.model_type == "GP":
            self.default_config = GP_CONFIG
        elif self.model_type == "delUQ":
            self.default_config = DELUQ_CONFIG
        elif self.model_type == 'nnEnsemble':
            self.default_config = NNENSEMBLE_CONFIG
        else:
            raise ValueError(f"The model type {self.model_type} has not yet been implemented. Options: 'GP', 'delUQ', or 'nnEnsemble'")

        self.config_settings = {"model_type": self.model_type}
        self.load_config()

    def __copy__(self):
        """
        A magic method to allow this class to be copied with copy(instance_of_ConfigSettings).
        """
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __contains__(self, key: str):
        """
        A magic method to allow us to see if certain keys are in our configuration.
        This can be called with the 'in' keyword.

        :param key: The configuration option to look for in our configuration settings.
        :returns: True if `key` is in our configuration settings already. False otherwise.
        """
        return key in self.config_settings

    def _verify_config_settings(self):
        """
        Ensure all required configuration values are set. If they're not,
        set them to their default values.
        """
        for required_config_key, default_config_val in self.default_config.items():
            # If this raises a KeyError it means the key isn't in the current config_settings
            try:
                self.config_settings[required_config_key]
            except KeyError:
                self.config_settings[required_config_key] = default_config_val

    def load_config(self):
        """
        Load in all of the configuration options provided by the user.
        """
        config = None
        if self.config_file is not None:
            with open(self.config_file, "r") as file:
                if self.config_file.endswith(".yaml"):
                    config = yaml.safe_load(file)
                elif self.config_file.endswith(".json"):
                    config = json.load(file)
                else:
                    raise ValueError(f"The config file {self.config_file} must be either a yaml file or a json file.")

        if config is not None:
            self.config_settings.update(config)
        self._verify_config_settings()

    def get_setting(self, setting_name: str) -> Any:
        """
        Return the setting associated with `setting_name`.

        :param setting_name: The name of the setting to get the value of
        :returns: The value of the setting `setting_name`
        """
        try:
            return self.config_settings[setting_name]
        except KeyError as exc:
            raise KeyError(f"The setting {setting_name} is not a valid setting.") from exc

    def set_setting(self, setting_name: str, setting_val: Any):
        """
        Set a new configuration setting option.

        :param setting_name: The name of the setting to set
        :param setting_val: The value of the new configuration setting
        """
        self.config_settings[setting_name] = setting_val
