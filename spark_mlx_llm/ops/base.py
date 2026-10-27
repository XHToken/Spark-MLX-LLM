"""Shared validation for operator-owned configuration dataclasses.

Operator functions deliberately keep their own signatures. Configuration
mixins share field discovery and validation without a runtime registry.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields

__all__ = ["OperatorConfig"]


@dataclass(kw_only=True)
class OperatorConfig:
    """Keyword-only fields can also be inherited by the model's flat config."""

    def __post_init__(self):
        self.validate()

    @classmethod
    def from_dict(cls, values: Mapping):
        """Extract this operator's fields from a checkpoint/model config."""
        return cls(
            **{
                item.name: values[item.name]
                for item in fields(cls)
                if item.name in values
            }
        )

    @classmethod
    def choices(cls, name: str) -> tuple:
        """Use the same choices for validation and CLI argument definitions."""
        item = next(item for item in fields(cls) if item.name == name)
        return item.metadata["choices"]

    @classmethod
    def validate_values(cls, values: Mapping):
        """Validate supplied fields without filling missing checkpoint values."""
        for item in fields(cls):
            if item.name not in values:
                continue
            value = values[item.name]
            if type(item.default) is bool and type(value) is not bool:
                raise ValueError(f"{item.name} must be a boolean, got {value!r}")
            choices = item.metadata.get("choices")
            if choices is not None and value not in choices:
                raise ValueError(f"{item.name} must be one of {choices}, got {value!r}")

    def validate(self):
        self.validate_values(
            {item.name: getattr(self, item.name) for item in fields(self)}
        )
