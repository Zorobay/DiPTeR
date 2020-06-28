import inspect
import logging
import time
import typing
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from glumpy import gloo
from node_graph.data_type import DataType
from src.misc import string_funcs, render_funcs
from src.shaders.parsing.parsing import GLSLCode, preprocess_imports
from torch import Tensor

_logger = logging.getLogger(__name__)

# Define useful constants for numerical manipulation
TINY_FLOAT = torch.finfo(torch.float32).eps

# GLSL Shaders Directory
GLSL_SHADERS_DIR = Path(__file__) / ".." / ".." / ".." / "res" / "shaders"


def read_glsl_source_file(filename) -> typing.List[str]:
    filepath = GLSL_SHADERS_DIR / filename
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


def connect_code(node: 'GShaderNode', code: GLSLCode):
    """
    Recursively connects the other_code to all other connected other_code in the NodeGraph tracked by 'node'.
    :param node: Node the holds the GLSLCode in 'other_code'
    :param code: The parsed shader code that is held by the Node 'node'
    """
    code.reset(node.get_num())
    for socket in node.get_in_sockets():
        assert socket.type() == socket.INPUT

        socket_arg = socket.label()

        connected_nodes = socket.get_connected_nodes()
        assert len(connected_nodes) <= 1

        if len(connected_nodes) > 0:
            connected_node = connected_nodes[0]
            connected_code = connected_node.get_shader().get_code()

            # Recurse...
            connect_code(connected_node, connected_code)

            code.connect(socket_arg, connected_code)


class ShaderInput:

    def __init__(self, display_label: str, argument: str, dtype: DataType, range_: typing.Tuple[float, float], default: typing.Any):
        self._display_label = display_label
        self._argument = argument
        self._dtype = dtype
        self._range = range_
        self._default = default

    def get_display_label(self) -> str:
        """Returns the formatted display label for this input."""
        return self._display_label

    def get_argument(self) -> str:
        """Returns the name of the argument of this input as it appears in the shader definition."""
        return self._argument

    def dtype(self):
        return self._dtype

    def get_range(self) -> typing.Tuple[float, float]:
        """Returns the range of value that this input can accept as a tuple of (min,max)."""
        return self._range

    def get_default(self):
        """Returns the default value of this input."""
        return self._default


class ShaderOutput:

    def __init__(self, display_label: str, dtype: DataType):
        self._display_label = display_label
        self._dtype = dtype

    def get_display_label(self) -> str:
        """Returns the formatted display label for this input."""
        return self._display_label

    def dtype(self):
        return self._dtype


class Shader(ABC):
    FRAGMENT_SHADER_FUNCTION = None
    FRAGMENT_SHADER_FILENAME = None
    frag_pos = None
    width = 100
    height = 100

    def __init__(self):
        self.code = None
        self._parsed_code = None

        self._parse()

    def get_size(self):
        print("{}: {}".format(self.__class__, Shader.width))

    def _parse(self):
        assert self.__class__.FRAGMENT_SHADER_FILENAME, "Name of GLSL fragment shader need to be set by subclass to FRAGMENT_SHADER_FILENAME " \
                                                        "constant."
        assert self.__class__.FRAGMENT_SHADER_FUNCTION, "Name of GLSL fragment shader primary function need to be set by subclass {} to " \
                                                        "FRAGMENT_SHADER_FUNCTION constant.".format(self.__class__)

        self.code = read_glsl_source_file(self.FRAGMENT_SHADER_FILENAME)
        self._parsed_code = GLSLCode(self.code, self.FRAGMENT_SHADER_FILENAME, self.FRAGMENT_SHADER_FUNCTION)

    def get_parsed_code(self) -> GLSLCode:
        return self._parsed_code

    def get_input_by_arg(self, argument: str) -> typing.Union[None, ShaderInput]:
        for inp in self.get_inputs():
            if inp.get_argument() == argument:
                return inp

        return None

    @classmethod
    def set_render_size(cls, width: int, height: int):
        Shader.width = width
        Shader.height = height
        Shader.frag_pos = render_funcs.generate_frag_pos(width, height)

    @abstractmethod
    def get_inputs(self) -> typing.List[ShaderInput]:
        """Returns a list of ShaderInput objects that describe the inputs of this shader."""

    # TODO make abstract (for now this is true for all shaders
    def get_outputs(self) -> typing.List[ShaderOutput]:
        """Returns a list of ShaderOutput objects that describe the outputs of this shader."""
        return [
            ShaderOutput("Color", DataType.Vec3_RGB)
        ]

    @abstractmethod
    def shade_mat(self, **args) -> Tensor:
        pass

    def get_name(self) -> str:
        """Returns the formatted title of this class"""
        return " ".join(string_funcs.split_on_upper_case(type(self).__name__))


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

        py_args = [inp.get_argument() for inp in self.get_inputs()]
        shade_mat_args = get_function_arguments(self.shade_mat)
        glsl_args = func.arguments

        # First, check that frag_pos is the first argument of the functions
        assert glsl_args[0].type == "vec3", "Type of glsl shade primary_function's first argument should be vec3, not {}".format(glsl_args[0].type)

        for py_arg, shade_mat_arg, glsl_arg in zip(py_args, shade_mat_args, glsl_args[1:]):
            assert py_arg == shade_mat_arg, "Name of Python input argument and argument to Python matrix shade function do not coincide: {} != {}" \
                .format(py_arg, shade_mat_arg)
            assert py_arg == glsl_arg.name, "Name of Python input argument and argument to GLSL shade function do not coincide: {} != {}" \
                .format(py_arg, glsl_arg.name)

        _logger.debug("Verified arguments of class {}.".format(self.__class__))

        self.__class__.verified = True

    def get_code(self) -> GLSLCode:
        return self._parsed_code


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

    def compile_fragment_shader(self, node: 'GShaderNode') -> gloo.FragmentShader:
        """
        Constructs the fragment shader other_code from this shader and (optionally) all connected shaders and compiles it.

        :param node: The Node object that represents this shader in the node node_graph. If this is not provided, this function will only compile the
        code belonging to this shader.
        :return: the compiled FragmentShader
        """
        assert self.__class__.FRAGMENT_SHADER_FILENAME, "Name of GLSL fragment shader need to be set by subclass"

        if not node:
            fragment_code = read_glsl_source_file(self.FRAGMENT_SHADER_FILENAME)
            fragment_code = preprocess_imports(fragment_code)
            return gloo.FragmentShader(fragment_code, version=self._glsl_version)
        else:
            output_code = self._parsed_code
            connect_code(node, output_code)
            generated_code = output_code.generate_code()
            return gloo.FragmentShader(generated_code, version=self._glsl_version)

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
        for inp in self.get_inputs():
            self._program[inp.get_argument()] = inputs[i].detach()
            i += 1

    def get_program(self) -> gloo.Program:
        return self._program

    def get_parameters_list(self, requires_grad=False, randomize=False):

        params = []

        for inp in self.get_inputs():
            val = self._program[inp.get_argument()]
            ran = inp.get_range()
            if "vec" not in inp.dtype():
                val = np.array(val[0])  # Uniforms are stored in array even if they're single floats

            if randomize:
                tensor = torch.from_numpy(np.random.uniform(ran[0], ran[1], size=val.size))
            else:
                tensor = torch.from_numpy(val)

            tensor = tensor.float()
            tensor.requires_grad = requires_grad
            params.append(tensor)

        return params

    def compile(self, node: 'GShaderNode' = None) -> typing.Tuple[gloo.VertexShader, gloo.FragmentShader, gloo.Program]:
        """
        Compiles the full program with vertex and fragment shader and returns a tuple with the VertexShader, FragmentShader and compiled
        Program. Does **not** save the new shader program internally.

        :param node: The GShaderNode object that contains this Shader.
        :return: a tuple with the compiled VertexShader, FragmentShader and Program
        """
        start = time.time()
        vertex_shader = self.compile_vertex_shader()
        fragment_shader = self.compile_fragment_shader(node)
        program = gloo.Program(vertex_shader, fragment_shader, count=0, version=self._glsl_version)
        compile_time = time.time() - start

        _logger.debug("Compiled {} in {:.6f}ms".format(self.get_name(), compile_time * 1000))

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
            for inp in self.get_inputs():
                default = inp.get_default()
                if isinstance(default, Tensor):
                    default = default.numpy()

                program[inp.get_argument()] = default
