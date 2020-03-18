import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path

from glumpy import gloo

from src.misc import string_funcs
from src.opengl.shader_types import INTERNAL_TYPE_RGBA
from src.shaders.lib.glsl_builtins import *

_logger = logging.getLogger(__name__)

GLSL_INCLUDE_DIR = "res/lib/"
REG_GLSL_IMPORT = re.compile(r"\s*#\s*import\s*[\"'](\w*\.?\w*)[\"']", flags=re.ASCII)


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

    def __init__(self):
        self.VERTEX_SHADER_FILENAME = None
        self.FRAGMENT_SHADER_FILENAME = None
        self._program = None
        self._vertex_shader = None
        self._fragment_shader = None
        self._glsl_version = "430"

    def get_vertex_shader(self) -> gloo.VertexShader:
        """Compiles the vertex shader code associated with this shader and returns it in a glumpy VertexShader object."""
        if not self._vertex_shader:
            assert self.VERTEX_SHADER_FILENAME, "Name of GLSL vertex shader need to be set by subclass"
            vertex_shader_path = Path(Path.cwd() / "res" / self.VERTEX_SHADER_FILENAME)

            try:
                with open(vertex_shader_path, "r") as v:
                    vertex_code = preprocess_imports(v.readlines())
            except FileNotFoundError as e:
                _logger.error("Could not find vertex shader at path %s.", vertex_shader_path)
                raise e

            self._vertex_shader = gloo.VertexShader(vertex_code, version=self._glsl_version)

        return self._vertex_shader

    def get_fragment_shader(self) -> gloo.FragmentShader:
        """Compiles the fragment shader code associated with this shader and returns it in a glumpy FragmentShader object."""
        if not self._fragment_shader:
            assert self.FRAGMENT_SHADER_FILENAME, "Name of GLSL fragment shader need to be set by subclass"
            fragment_shader_path = Path(Path.cwd() / "res" / self.FRAGMENT_SHADER_FILENAME)

            try:
                with open(fragment_shader_path, "r") as f:
                    fragment_code = preprocess_imports(f.readlines())
            except FileNotFoundError as e:
                _logger.error("Could not find fragment shader at path %s.", fragment_shader_path)
                raise e

            self._fragment_shader = gloo.FragmentShader(fragment_code, version=self._glsl_version)

        return self._fragment_shader

    def get_program(self, vertex_count: int = 0, set_defaults: bool = False) -> gloo.Program:
        """Compiles the full program with vertex and fragment shader and returns it in a glumpy Program object"""
        vertex_shader = self.get_vertex_shader()
        fragment_shader = self.get_fragment_shader()
        self._program = gloo.Program(vertex_shader, fragment_shader, count=vertex_count, version=self._glsl_version)

        if set_defaults:
            for _, uniform, _, _, default in self.get_inputs():
                self._program[uniform] = default

        return self._program

    def get_name(self) -> str:
        """Returns the formatted name of this class"""
        return " ".join(string_funcs.split_on_upper_case(type(self).__name__))

    @abstractmethod
    def get_inputs(self) -> typing.List[typing.Tuple[str, str, str, typing.Tuple[float, float], typing.Any]]:
        """Returns a list of tuples with information about the input parameters of this shader.

            :return: a tuple on the form (formatted name, uniform name, internal type, (min value, max value), default value)
        """

    # TODO make abstract
    def get_outputs(self):
        """Returns a list of tuples with information about the output parameters of this shader.

            :return: a tuple on the form (formatted name, internal type)
        """
        return [
            ("Color", INTERNAL_TYPE_RGBA)
        ]

    @abstractmethod
    def shade(self, *args) -> ndarray:
        pass
