import torch

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
        self._is_normalized = False

    def __str__(self):
        return "Parameter({})".format(self._t)

    @torch.no_grad()
    def get_normalized(self, index=-1) -> torch.Tensor:
        """Returns the value of this Parameter normalized to the interval [0,1]."""
        return (self.get_value(index) - self.get_min()) / (self.get_max() - self.get_min())

    @torch.no_grad()
    def normalize(self):
        if not self._is_normalized:
            (self._t.sub_(self.get_min())).div_(self.get_max()-self.get_min())
            self._is_normalized = True

    @torch.no_grad()
    def unnormalize(self):
        if self._is_normalized:
            (self._t.mul_(self.get_max()-self.get_min())).add_(self.get_min())
            self._is_normalized = False

    def tensor(self) -> torch.Tensor:
        return self._t

    def get_min(self):
        """Returns the specified minimum limit of this Parameter."""
        return self.get_limits()[0]

    def get_max(self):
        """Returns the specified maximum limit of this Parameter."""
        return self.get_limits()[1]

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
