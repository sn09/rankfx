"""Module with basic config implementation."""

from __future__ import annotations

import pathlib
from dataclasses import asdict
from typing import Any

import yaml
from pydantic.dataclasses import ConfigDict, dataclass


@dataclass
class BaseConfig:
    """Base config implementation."""
    model_config = ConfigDict(extra="ignore")

    def as_dict(self) -> dict[str, Any]:
        """Get config dict representation."""
        return asdict(self)

    @staticmethod
    def from_yaml(filepath: str | pathlib.Path) -> BaseConfig:
        """Load config from yaml file.

        Args:
            filepath: path to yaml file

        Returns:
            Config instance
        """
        with open(filepath) as fin:
            content = yaml.safe_load(fin)
        return BaseConfig(**content)

    def to_yaml(self) -> str:
        """Export config to yaml format.

        Returns:
            Config in yaml string format
        """
        return yaml.dump(self.as_dict())
