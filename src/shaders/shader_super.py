import inspect
import logging
import time
import typing
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from glumpy import gloo
from torch import Tensor

from src.gui.node_editor.socket import Socket
from src.misc import string_funcs, render_funcs
from src.opengl.internal_types import *
from src.shaders.parsing.parsing import GLSLCode, preprocess_imports

_logger = logging.getLogger(__name__)

# Define useful constants for numerical manipulation
TINY_FLOAT = torch.finfo(torch.float32).eps


def read_glsl_source_file(filename) -> typing.List[str]:
    filepath = Path.cwd() / "res" / filename
    try:
        with open(filepath, "r") as v:
            return v.readlines()
    except FileNotFoundError as e:
        _logger.error("Could not find file at path %s.", filepath)
        raise e
    except IOError as e:
        _logger.error("Could not read file at path %s.", filepath)
        raise e


def get_function_arguments(func: typing.Callable):
    return list(inspect.signature(func).parameters.keys())


def connect_code(node: 'Node', code: GLSLCode):
    """
    Recursively connects the other_code to all other connected other_code in the NodeGraph tracked by 'node'.
    :param node: Node the holds the GLSLCode in 'other_code'
    :param code: The parsed shader code that is held by the Node 'node'
    """
    code.reset(node.get_num())
    for _, socket in node.get_in_sockets().items():
        assert socket.socket_type == Socket.SOCKET_INPUT

        socket_arg = socket.get_argument()

        connected_nodes = socket.get_connected_nodes()
        assert len(connected_nodes) <= 1

        if len(connected_nodes) > 0:
            connected_node = connected_nodes[0]
            connected_code = connected_node.get_shader().get_code()

            # Recurse...
            connect_code(connected_node, connected_code)

            code.connect(socket_arg, connected_code)


class Shader(ABC):
    FRAGMENT_SHADER_FUNCTION = None
    FRAGMENT_SHADER_FILENAME = None
    vert_pos = None
    width = 100
    height = 100

    def __init__(self):
        self.code = None
        self._parsed_code = None

        self._parse()

    def _parse(self):
        assert self.__class__.FRAGMENT_SHADER_FILENAME, "Name of GLSL fragment shader need to be set by subclass to FRAGMENT_SHADER_FILENAME " \
                                                        "constant."
        assert self.__class__.FRAGMENT_SHADER_FUNCTION, "Name of GLSL fragment shader primary function need to be set by subclass {} to " \
                                                        "FRAGMENT_SHADER_FUNCTION constant.".format(self.__class__)

        self.code = read_glsl_source_file(self.FRAGMENT_SHADER_FILENAME)
        self._parsed_code = GLSLCode(self.code, self.FRAGMENT_SHADER_FILENAME, self.FRAGMENT_SHADER_FUNCTION)

    def get_parsed_code(self) -> GLSLCode:
        return self._parsed_code

    @classmethod
    def set_render_size(cls, width: int, height: int):
        cls.width = width
        cls.height = height
        cls.vert_pos = render_funcs.generate_vert_pos(width, height)

    @abstractmethod
    def get_inputs(self) -> typing.List[typing.Tuple[str, str, str, typing.Tuple[float, float], Tensor]]:
        """Returns a list of tuples with information about the widgets parameters of this shader.

            :return: a tuple on the form (formatted title, argument title, internal type, (min value, max value), default value)
        """

    # TODO make abstract (for now this is true for all shaders
    def get_outputs(self) -> typing.List[typing.Tuple[str, str]]:
        """Returns a list of tuples with information about the output parameters of this shader.

            :return: a tuple on the form (formatted title, internal type)
        """
        return [
            ("Color", INTERNAL_TYPE_ARRAY_RGB)
        ]

    def shade(self, vert_pos: Tensor, *args) -> Tensor:
        pass

    @abstractmethod
    def shade_mat(self, **args) -> Tensor:
        pass


class FunctionShader(Shader, ABC):
    """A shader that is represented by only a single function in GLSL, and can thus on its own not be made into a program."""
    FRAGMENT_SHADER_FILENAME = None
    verified = False

    def __init__(self):
        super(FunctionShader, self).__init__()
        self._verify_arguments()

    def _verify_arguments(self):
        """Checks that arguments of the GLSL implementation and python implementation coincide."""
        if self.__class__.verified:
            return
        func = self._parsed_code.get_primary_function()

        py_args = [inp[1] for inp in self.get_inputs()]
        shade_args = get_function_arguments(self.shade)
        shade_mat_args = get_function_arguments(self.shade_mat)
        glsl_args = func.arguments

        # First, check that vert_pos is the first argument of the functions
        assert glsl_args[0].type == "vec3", "Type of glsl shade primary_function's first argument should be vec3, not {}".format(glsl_args[0].type)
        assert glsl_args[0].name == shade_args[0], \
            "Name of first argument of shader functions (both in python and GLSL) should be vert_pos, not {}".format(glsl_args[0].name)

        for py_arg, shade_arg, shade_mat_arg, glsl_arg in zip(py_args, shade_args[1:], shade_mat_args, glsl_args[1:]):
            assert py_arg == shade_arg, "Name of Python input argument and argument to Python shade function do not coincide: {} != {}" \
                .format(py_arg, shade_arg)
            assert py_arg == shade_mat_arg, "Name of Python input argument and argument to Python matrix shade function do not coincide: {} != {}" \
                .format(py_arg, shade_mat_arg)
            assert py_arg == glsl_arg.name, "Name of Python input argument and argument to GLSL shade function do not coincide: {} != {}" \
                .format(py_arg, glsl_arg.name)

        _logger.debug("Verified arguments of class {}.".format(self.__class__))

        self.__class__.verified = True

    def get_code(self) -> GLSLCode:
        return self._parsed_code

    def get_name(self) -> str:
        """Returns the formatted title of this class"""
        return " ".join(string_funcs.split_on_upper_case(type(self).__name__))

    def get_parameters_dict(self) -> typing.Dict[str, typing.Any]:
        params = {}
        for _, arg, _, _, _ in self.get_inputs():
            params[arg] = self._program[arg]

        return params


class CompilableShader(Shader, ABC):
    VERTEX_SHADER_FILENAME = "vertex_shader.glsl"
    FRAGMENT_SHADER_FUNCTION = "main"

    def __init__(self):
        super().__init__()
        self._program = None
        self._vertex_shader = None
        self._fragment_shader = None
        self._glsl_version = "430"

        self._set_program()

    def compile_vertex_shader(self) -> gloo.VertexShader:
        """Compiles the vertex shader other_code associated with this shader and returns it in a glumpy VertexShader object."""
        vertex_code = preprocess_imports(read_glsl_source_file(self.VERTEX_SHADER_FILENAME))
        return gloo.VertexShader(vertex_code, version=self._glsl_version)

    def compile_fragment_shader(self, node: 'Node') -> gloo.FragmentShader:
        """
        Constructs the fragment shader other_code from this shader and (optionally) all connected shaders and compiles it.

        :param node: The Node object that represents this shader in the node graph. If this is not provided, this function will only compile the
        code belonging to this shader.
        :return: the compiled FragmentShader
        """
        assert self.__class__.FRAGMENT_SHADER_FILENAME, "Name of GLSL fragment shader need to be set by subclass"

        start_time = time.time()

        if not node:
            fragment_code = read_glsl_source_file(self.FRAGMENT_SHADER_FILENAME)
            fragment_code = preprocess_imports(fragment_code)
            self._fragment_shader = gloo.FragmentShader(fragment_code, version=self._glsl_version)
            return self._fragment_shader
        else:
            output_code = self._parsed_code
            connect_code(node, output_code)
            generated_code = output_code.generate_code()
            self._fragment_shader = gloo.FragmentShader(generated_code, version=self._glsl_version)

        compile_time = time.time() - start_time
        _logger.debug("Recompiled {} in {:.6f}ms".format(node.get_title(), compile_time * 1000))

        return self._fragment_shader

    def set_input_by_uniform(self, uniform: str, value: typing.Any):
        """
        Set the value of a uniform of the Fragment Shader in the program held by this Shader. Raises an AttributeError if the uniform is not valid.
        :param uniform: the title of uniform. Must match the uniform defined in the GLSL shader.
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
        for _, arg, _, _, _ in self.get_inputs():
            self._program[arg] = inputs[i].detach()
            i += 1

    def get_program(self) -> gloo.Program:
        return self._program

    def get_parameters_list(self, requires_grad=False, randomize=False):

        params = []

        for _, arg, internal_type, ran, _ in self.get_inputs():
            val = self._program[arg]
            if "array" not in internal_type:
                val = np.array(val[0])  # Uniforms are stored in array even if they're single floats

            if randomize:
                tensor = torch.from_numpy(np.random.uniform(ran[0], ran[1], size=val.size))
            else:
                tensor = torch.from_numpy(val)

            tensor = tensor.float()
            tensor.requires_grad = requires_grad
            params.append(tensor)

        return params

    def compile(self, node: 'Node' = None) -> typing.Tuple[gloo.VertexShader, gloo.FragmentShader, gloo.Program]:
        """
        Compiles the full program with vertex and fragment shader and returns a tuple with the VertexShader, FragmentShader and compiled
        Program. Does **not** save the new shader program internally.

        :param node: The Node object that contains this Shader.
        :return: a tuple with the compiled VertexShader, FragmentShader and Program
        """
        print("Compile called!")
        vertex_shader = self.compile_vertex_shader()
        fragment_shader = self.compile_fragment_shader(node)

        program = gloo.Program(vertex_shader, fragment_shader, count=0, version=self._glsl_version)

        return vertex_shader, fragment_shader, program

    def recompile(self, node: 'Node' = None):
        """
        Compiles the full program with vertex and fragment shader and returns a tuple with the VertexShader, FragmentShader and compiled
        Program. Saves the new compiled shader program internally.

        :param node: The Node object that contains this Shader.
        """
        self._vertex_shader, self._fragment_shader, self._program = self.compile(node)

    def _set_program(self):
        self._vertex_shader, self._fragment_shader, self._program = self.compile()
        self.set_defaults()

    def set_defaults(self, program: gloo.Program = None):
        program = self._program if program is None else program

        if program is not None:
            for _, arg, type_, _, default in self.get_inputs():

                if isinstance(default, Tensor):
                    default = default.numpy()

                program[arg] = default
