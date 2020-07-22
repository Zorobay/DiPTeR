import torch
import numpy as np
from dipter.node_graph.data_type import is_vector
from dipter.shaders.shader_io import ShaderInputParameter


class Parameter(ShaderInputParameter):
    """A class to represent a PyTorch tensor with extra metadata and value restrictions specified by a ShaderInput."""

    def __init__(self, si: ShaderInputParameter, value: torch.Tensor):
        super().__init__(si.get_display_label(), si.get_argument(), si.dtype(), si.get_limits(),
                         si.get_default(), connectable=si.is_connectable(), force_scalar=si.is_scalar(),
                         names=si.get_names())
        self._t = value
        self._data = None
        self._mod_arg = None

    def __str__(self):
        return "Parameter({})".format(self._t)

    def randomize(self):
        """Assigns a random value within the specified limits to this Parameter."""
        random_value = np.random.uniform(self.get_min(), self.get_max(), size=self._t.shape)
        self.set_value(np.round(random_value, decimals=3))

    def tensor(self) -> torch.Tensor:
        return self._t

    def set_default(self):
        """Set the value to the default value."""
        self._t.data = torch.as_tensor(self.get_default())

    def set_modified_arg(self, mod_arg: str):
        """Set the modified arg of this Parameter."""
        self._mod_arg = mod_arg

    def get_modified_arg(self) -> str:
        """Get the modified argument of this Parameter if set, otherwise returns None."""
        return self._mod_arg

    def save_value(self):
        """Saves the current value of this Parameter. Can be restored by calling 'restore_value()'."""
        self._data = self._t.data.clone()

    def restore_value(self):
        """Restores the data of the contained tensor to the value of the last saved data."""
        self._t.data = self._data

    def set_value(self, value, index=-1):
        if index >= 0:
            self._t[index] = torch.as_tensor(value)
        else:
            self._t.data = torch.as_tensor(value)

    def get_value(self, index=-1):
        if index >= 0:
            return self._t[index].detach().clone().cpu().numpy()
        else:
            return self._t.detach().clone().cpu().numpy()

    def is_vector(self) -> bool:
        return is_vector(self.dtype())

    def shape(self) -> torch.Size:
        return self._t.shape
