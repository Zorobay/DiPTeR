import torch
from node_graph.data_type import is_vector, dtype_size
from shaders.shader_super import ShaderInput


class Parameter(ShaderInput):
    """A class to represent a PyTorch tensor with extra metadata and value restrictions specified by a ShaderInput."""

    def __init__(self, shader_input: ShaderInput, value: torch.Tensor):
        super().__init__(shader_input.get_display_label(), shader_input.get_argument(), shader_input.dtype(), shader_input.get_range(),
                         shader_input.get_default())
        self._t = value

    def __str__(self):
        return "Parameter({})".format(self._t)

    def tensor(self) -> torch.Tensor:
        return self._t

    def set_default(self):
        """Set the value to the default value."""
        self._t.data = torch.as_tensor(self.get_default())

    def is_vector(self) -> bool:
        return is_vector(self.dtype())

    def shape(self) -> torch.Size:
        return self._t.shape
