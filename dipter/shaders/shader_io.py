import typing

import numpy as np

from dipter.node_graph.data_type import DataType


class ShaderInputParameter:

    def __init__(self, display_label: str, argument: str, dtype: DataType, limits: typing.Tuple[float, float], default: typing.Any,
                 connectable: bool = True, force_scalar: bool = False, names: typing.List[str] = None):
        self._display_label = display_label
        self._argument = argument
        self._dtype = dtype
        self._limits = limits
        self._default = default
        self._connectable = connectable
        self._force_scalar = force_scalar
        if names is None:
            names = list()
        self._names = names

        if self._force_scalar:
            self._connectable = False

    def get_min(self):
        """Returns the specified minimum limit of this Parameter."""
        return self.get_limits()[0]

    def get_max(self):
        """Returns the specified maximum limit of this Parameter."""
        return self.get_limits()[1]

    def get_range(self) -> float:
        """Returns the range of the limit."""
        return self.get_max() - self.get_min()

    def get_random_value(self):
        """Returns a random value within the limits of the correct shape for this input parameter."""
        low = self.get_min()
        high = self.get_max()
        random = np.random.uniform(low, high, size=np.shape(self.get_default()))
        return np.round(random, decimals=3)

    def get_centered_random_value(self, spread: float = None):
        """Returns a random value within the limits of the correct shape that is centered around the default value."""
        if spread is None:
            spread = self.get_range() * 0.2

        center = self.get_default()
        center = np.where(center - spread < self.get_min(), center + spread, center)
        # if center-spread < self.get_min():
        #     center = center + spread

        random = np.clip(np.random.normal(center, scale=spread), a_min=self.get_min(), a_max=self.get_max())
        return np.round(random, decimals=3)

    def get_display_label(self) -> str:
        """Returns the formatted display label for this input."""
        return self._display_label

    def get_argument(self) -> str:
        """Returns the name of the argument of this input as it appears in the shader definition."""
        return self._argument

    def dtype(self) -> DataType:
        return self._dtype

    def get_limits(self) -> typing.Tuple[float, float]:
        """Returns the range of value that this input can accept as a tuple of (min,max)."""
        return self._limits

    def get_default(self):
        """Returns the default value of this input."""
        return self._default

    def is_connectable(self) -> bool:
        """Returns a boolean indicating whether this input can be connected to other nodes."""
        return self._connectable

    def is_scalar(self) -> bool:
        """Returns whether this parameter should be handled as a scalar, and thus not converted to matrix form when shading."""
        return self._force_scalar

    def get_names(self) -> typing.List[str]:
        """Returns a list of names of the values of this Input."""
        return self._names


class ShaderOutputParameter:

    def __init__(self, display_label: str, dtype: DataType, argument: str = None):
        self._display_label = display_label
        self._dtype = dtype
        self._argument = argument

    def get_display_label(self) -> str:
        """Returns the formatted display label for this input."""
        return self._display_label

    def dtype(self):
        return self._dtype

    def get_argument(self) -> str:
        return self._argument
