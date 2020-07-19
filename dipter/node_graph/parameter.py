import torch
from dipter.node_graph.data_type import is_vector
from dipter.shaders.shader_super import ShaderInputParameter


class Parameter(ShaderInputParameter):
    """A class to represent a PyTorch tensor with extra metadata and value restrictions specified by a ShaderInput."""

    def __init__(self, shader_input: ShaderInputParameter, value: torch.Tensor):
        super().__init__(shader_input.get_display_label(), shader_input.get_argument(), shader_input.dtype(), shader_input.get_limits(),
                         shader_input.get_default())
        self._t = value
        self._data = None
        self._mod_arg = None

    def __str__(self):
        return "Parameter({})".format(self._t)

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
