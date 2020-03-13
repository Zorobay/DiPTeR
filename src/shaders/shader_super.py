import logging
import typing
from pathlib import Path

from glumpy import gloo

from src.misc import string_funcs
from src.opengl.shader_types import from_gl_type, Type

_logger = logging.getLogger(__name__)


class Shader:

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
                    vertex_code = v.read()
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
                    fragment_code = f.read()
            except FileNotFoundError as e:
                _logger.error("Could not find fragment shader at path %s.", fragment_shader_path)
                raise e

            self._fragment_shader = gloo.FragmentShader(fragment_code, version=self._glsl_version)

        return self._fragment_shader

    def get_program(self, vertex_count: int) -> gloo.Program:
        """Compiles the full program with vertex and fragment shader and returns it in a glumpy Program object"""
        vertex_shader = self.get_vertex_shader()
        fragment_shader = self.get_fragment_shader()
        self._program = gloo.Program(vertex_shader, fragment_shader, count=vertex_count, version=self._glsl_version)
        return self._program

    def get_parameters(self) -> typing.List[typing.Tuple[str, Type]]:
        """Find the uniforms defined in the fragment shader and returns their formatted name and Type.

            :return: A list of (param name, param Type) tuples
        """

        python_uniforms = []
        uniforms = self.get_fragment_shader().uniforms

        for n, gl_type in uniforms:
            formatted_name = " ".join(string_funcs.snake_case_to_names(n))
            type_ = from_gl_type(gl_type)
            python_uniforms.append((formatted_name, type_))

        return python_uniforms

    def get_name(self) -> str:
        """Returns the formatted name of this class"""
        return " ".join(string_funcs.split_on_upper_case(type(self).__name__))
