import inspect
import logging
import os
import time
import typing
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from glumpy import gloo
from dipter.node_graph.data_type import DataType
from dipter.misc import string_funcs, render_funcs
from dipter.shaders.parsing.parsing import GLSLCode, preprocess_imports
from torch import Tensor

_logger = logging.getLogger(__name__)

# Define useful constants for numerical manipulation
TORCH_FLOAT_FINFO = torch.finfo(torch.float32)
TINY_FLOAT = TORCH_FLOAT_FINFO.eps
FLOAT_INTERVAL = (TORCH_FLOAT_FINFO.min,TORCH_FLOAT_FINFO.max)

# GLSL Shaders Directory
GLSL_SHADERS_DIR = Path(os.getcwd()) / "res" / "shaders"


def read_glsl_source_file(filename) -> typing.List[str]:
    filepath = GLSL_SHADERS_DIR / filename
    try:
        with open(filepath, "r") as v:
            return v.readlines()
    except NotADirectoryError as e:
        _logger.warning("Failed to read GLSL source file at %s", filepath)
        return []
    except FileNotFoundError as e:
        raise FileNotFoundError("Could not find file at path %s.", filepath)
    except IOError as e:
        raise IOError("Could not read file at path %s.", filepath)


def get_function_arguments(func: typing.Callable):
    return list(inspect.signature(func).parameters.keys())


def connect_code(node: 'GShaderNode', code: GLSLCode):
    """
    Recursively connects the other_code to all other connected other_code in the NodeGraph tracked by 'node'.
    :param node: Node the holds the GLSLCode in 'other_code'
    :param code: The parsed shader code that is held by the Node 'node'
    """
    code.reset(node.get_num())
    for socket in node.get_input_sockets():
        assert socket.type() == socket.INPUT

        socket_arg = socket.label()

        connected_nodes = socket.get_connected_nodes()
        assert len(connected_nodes) <= 1

        if len(connected_nodes) == 1:
            connected_node = connected_nodes[0]
            connected_code = connected_node.get_shader().get_code()

            connected_arg = None
            if connected_code.get_primary_function().is_inout():  # We will have to connect the 'out' argument too
                connected_socket = socket.get_connected_sockets()
                assert len(connected_socket) == 1
                connected_socket = connected_socket[0]
                shader_output = connected_node.get_shader().get_outputs()[connected_socket.get_index()]
                connected_arg = shader_output.get_argument()

            # Recurse...
            connect_code(connected_node, connected_code)

            code.connect(socket_arg, connected_code, out_arg=connected_arg)


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


class Shader:
    FRAGMENT_SHADER_FUNCTION = None
    FRAGMENT_SHADER_FILENAME = None
    _ren_width = 100
    _ren_height = 100
    _frag_pos_matrix = torch.empty((_ren_width, _ren_height))

    def __init__(self):
        self.code = None
        self._parsed_code = None

        self._parse()

    def _parse(self):
        assert self.__class__.FRAGMENT_SHADER_FILENAME, "Name of GLSL fragment shader need to be set by subclass to FRAGMENT_SHADER_FILENAME " \
                                                        "constant."
        assert self.__class__.FRAGMENT_SHADER_FUNCTION, "Name of GLSL fragment shader primary function need to be set by subclass {} to " \
                                                        "FRAGMENT_SHADER_FUNCTION constant.".format(self.__class__)

        try:
            self.code = read_glsl_source_file(self.FRAGMENT_SHADER_FILENAME)
        except Exception as e:
            _logger.warning("Failed to parse GLSL source file due to an error reading the file: %s", str(e))

        self._parsed_code = GLSLCode(self.code, self.FRAGMENT_SHADER_FILENAME, self.FRAGMENT_SHADER_FUNCTION)

    def get_parsed_code(self) -> GLSLCode:
        return self._parsed_code

    def get_input_by_arg(self, argument: str) -> typing.Union[None, ShaderInputParameter]:
        for inp in self.get_inputs():
            if inp.get_argument() == argument:
                return inp

        return None

    @classmethod
    def set_render_size(cls, width: int, height: int):
        Shader._ren_height = width
        Shader._ren_width = height
        Shader._frag_pos_matrix = render_funcs.generate_frag_pos(width, height)

    @classmethod
    def render_size(cls):
        return Shader._ren_width, Shader._ren_height

    @classmethod
    def render_width(cls):
        return Shader._ren_width

    @classmethod
    def render_height(cls):
        return Shader._ren_height

    @classmethod
    def frag_pos(cls):
        return Shader._frag_pos_matrix

    @abstractmethod
    def get_inputs(self) -> typing.List[ShaderInputParameter]:
        """Returns a list of ShaderInput objects that describe the inputs of this shader."""

    def get_outputs(self) -> typing.List[ShaderOutputParameter]:
        """Returns a list of ShaderOutput objects that describe the outputs of this shader."""
        return [
            ShaderOutputParameter("Color", DataType.Vec3_RGB)
        ]

    def shade(self, args: dict) -> Tensor:
        """Convenience function that converts all connectable arguments to matrix form before returning the rendered image."""
        width, height = Shader.render_width(), Shader.render_height()
        mat_args = dict()

        for key_arg, param in zip(args, self.get_inputs()):
            arg = args[key_arg]
            if (len(arg.shape) == 3 and arg.shape[0] == width and arg.shape[1] == height) or param.is_scalar():
                mat_args[key_arg] = arg.float()
            else:
                mat_args[key_arg] = arg.repeat(width, height, 1).float()

        return self.shade_mat(**mat_args)

    @abstractmethod
    def shade_mat(self, **args) -> Tensor:
        pass

    def get_name(self) -> str:
        """Returns the formatted title of this class"""
        return " ".join(string_funcs.split_on_upper_case(type(self).__name__))

    def tensor(self, data) -> Tensor:
        """Creates a matrix tensor of a shape that matches the rendering size from data. This is shorthand to creating tensors in the shade
        function."""
        return torch.tensor(data, dtype=torch.float32).repeat(*Shader.render_size(), 1)


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
            ran = inp.get_limits()
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
                    default = default.cpu().numpy()

                program[inp.get_argument()] = default
