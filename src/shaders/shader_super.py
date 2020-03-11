import logging
from pathlib import Path

from glumpy import gloo

_logger = logging.getLogger(__name__)


class Shader:

    def __init__(self):
        self.VERTEX_SHADER_FILENAME = None
        self.FRAGMENT_SHADER_FILENAME = None
        self._program = None

    def get_program(self, vertex_count: int):
        assert self.VERTEX_SHADER_FILENAME, "Name of GLSL vertex shader need to be set by subclass"
        assert self.FRAGMENT_SHADER_FILENAME, "Name of GLSL fragment shader need to be set by subclass"

        vertex_shader_path = Path(Path.cwd() / "res" / self.VERTEX_SHADER_FILENAME)
        fragment_shader_path = Path(Path.cwd() / "res" / self.FRAGMENT_SHADER_FILENAME)

        try:
            with open(vertex_shader_path, "r") as v:
                vertex_code = v.read()
        except FileNotFoundError as e:
            _logger.error("Could not find vertex shader at path %s.", vertex_shader_path)
            raise e

        try:
            with open(fragment_shader_path, "r") as f:
                fragment_code = f.read()
        except FileNotFoundError as e:
            _logger.error("Could not find fragment shader at path %s.", fragment_shader_path)
            raise e

        self._program = gloo.Program(vertex=vertex_code, fragment=fragment_code, count=vertex_count)
        return self._program
