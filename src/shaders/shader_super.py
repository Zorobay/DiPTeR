import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import torch

from glumpy import gloo
from glumpy.gloo import Program
from torch import Tensor

from src.misc import string_funcs
from src.opengl.shader_types import INTERNAL_TYPE_ARRAY_RGBA
from src.shaders.lib.glsl_builtins import *

_logger = logging.getLogger(__name__)

GLSL_INCLUDE_DIR = "res/lib/"
REG_GLSL_IMPORT = re.compile(r"\s*#\s*import\s*[\"'](\w*\.?\w*)[\"']", flags=re.ASCII)

# Define useful constants for numerical manipulation
TINY_FLOAT = anp.finfo(anp.float32).tiny


def preprocess_imports(code: typing.List[str]) -> str:
    lib_path = Path.cwd() / GLSL_INCLUDE_DIR
    for i, line in enumerate(code):
        match = REG_GLSL_IMPORT.match(line)
        if match:
            filename = match.group(1)
            with open(lib_path / filename, 'r') as f:
                libfile = f.read()
                code[i] = libfile  # Insert contents of libfile at line containing import statement

    return "".join(code)


class Shader(ABC):
    VERTEX_SHADER_FILENAME = None
    FRAGMENT_SHADER_FILENAME = None

    def __init__(self):
        self._program = None
        self._vertex_shader = None
        self._fragment_shader = None
        self._glsl_version = "430"

        # Load and set program
        self._set_program()

    def get_vertex_shader(self, new_instance=False) -> gloo.VertexShader:
        """Compiles the vertex shader code associated with this shader and returns it in a glumpy VertexShader object."""
        if not self._vertex_shader or new_instance:
            assert self.__class__.VERTEX_SHADER_FILENAME, "Name of GLSL vertex shader need to be set by subclass"
            vertex_shader_path = Path(Path.cwd() / "res" / self.VERTEX_SHADER_FILENAME)

            try:
                with open(vertex_shader_path, "r") as v:
                    vertex_code = preprocess_imports(v.readlines())
            except FileNotFoundError as e:
                _logger.error("Could not find vertex shader at path %s.", vertex_shader_path)
                raise e

            return gloo.VertexShader(vertex_code, version=self._glsl_version)

        return self._vertex_shader

    def get_fragment_shader(self, new_instance=False) -> gloo.FragmentShader:
        """Compiles the fragment shader code associated with this shader and returns it in a glumpy FragmentShader object."""
        if not self._fragment_shader or new_instance:
            assert self.__class__.FRAGMENT_SHADER_FILENAME, "Name of GLSL fragment shader need to be set by subclass"
            fragment_shader_path = Path(Path.cwd() / "res" / self.FRAGMENT_SHADER_FILENAME)

            try:
                with open(fragment_shader_path, "r") as f:
                    fragment_code = preprocess_imports(f.readlines())
            except FileNotFoundError as e:
                _logger.error("Could not find fragment shader at path %s.", fragment_shader_path)
                raise e

            return gloo.FragmentShader(fragment_code, version=self._glsl_version)

        return self._fragment_shader

    def get_program(self) -> gloo.Program:
        return self._program

    def get_program_copy(self):
        vertex_shader = self.get_vertex_shader(new_instance=True)
        fragment_shader = self.get_fragment_shader(new_instance=True)
        program = gloo.Program(vertex_shader, fragment_shader, count=0, version=self._glsl_version)

        # Copy uniforms
        uniforms = self._program.all_uniforms
        for u, _ in uniforms:
            program[u] = self._program[u]

        return program

    def _set_program(self):
        """Compiles the full program with vertex and fragment shader and returns it in a glumpy Program object"""
        self._vertex_shader = self.get_vertex_shader()
        self._fragment_shader = self.get_fragment_shader()
        self._program = gloo.Program(self._vertex_shader, self._fragment_shader, count=0, version=self._glsl_version)

        self.set_defaults()

    def set_defaults(self, program: Program = None):
        program = self._program if program is None else program

        if program is not None:
            for _, uniform, _, _, default in self.get_inputs():
                program[uniform] = default

    def set_input_by_uniform(self, uniform: str, value: typing.Any):
        """
        Set the value of a uniform of the Fragment Shader in the program held by this Shader. Raises an AttributeError if the uniform is not valid.
        :param uniform: the name of uniform. Must match the uniform defined in the GLSL shader.
        :param value: the new value of the uniform.
        """
        try:
            self._program[uniform] = value
        except AttributeError as e:
            _logger.error("Uniform %s does not exist in shader!", uniform)
            raise e

    def set_inputs(self, inputs):
        """
        Set the values of the inputs to this shader from a list of parameters. The length of the parameters must match the length of the list
        returned by get_inputs().
        :param inputs: a list of parameter values.
        """
        assert len(inputs) == len(self.get_inputs())
        i = 0
        for _, uniform, _, _, _ in self.get_inputs():
            self._program[uniform] = inputs[i]
            i += 1

    def get_name(self) -> str:
        """Returns the formatted name of this class"""
        return " ".join(string_funcs.split_on_upper_case(type(self).__name__))

    def get_parameters_dict(self) -> typing.Dict[str, typing.Any]:
        params = {}
        for _, uniform, _, _, _ in self.get_inputs():
            params[uniform] = self._program[uniform]

        return params

    def get_parameters_list(self) -> typing.List[typing.Any]:
        params = []

        for _, uniform, internal_type, _, _ in self.get_inputs():
            val = self._program[uniform]
            if not "array" in internal_type:
                val = val[0]  # Uniforms are stored in array even if they're single floats

            params.append(val)

        return params

    def get_parameters_list_torch(self, requires_grad=False):

        params = []

        for _, uniform, internal_type, _, _ in self.get_inputs():
            val = self._program[uniform]
            if not "array" in internal_type:
                val = val[0]  # Uniforms are stored in array even if they're single floats

            tensor = torch.from_numpy(np.array(val))
            tensor.requires_grad = requires_grad
            params.append(tensor)

        return params

    @abstractmethod
    def get_inputs(self) -> typing.List[typing.Tuple[str, str, str, typing.Tuple[float, float], typing.Any]]:
        """Returns a list of tuples with information about the widgets parameters of this shader.

            :return: a tuple on the form (formatted name, uniform name, internal type, (min value, max value), default value)
        """

    # TODO make abstract (for now this is true for all shaders
    def get_outputs(self) -> typing.List[typing.Tuple[str, str]]:
        """Returns a list of tuples with information about the output parameters of this shader.

            :return: a tuple on the form (formatted name, internal type)
        """
        return [
            ("Color", INTERNAL_TYPE_ARRAY_RGBA)
        ]

    @abstractmethod
    def shade(self, vert_pos: ndarray, *args) -> ndarray:
        pass
