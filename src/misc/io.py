import importlib
import typing


def import_class_from_string(string: str) -> typing.Type:
    """Returns a class at the end of an import string like 'src.shaders.material_output_shader.MaterialOutputShader'."""
    parts = string.split(".")
    modpath = ".".join(parts[0:-1])
    classname = parts[-1]
    module = importlib.import_module(modpath)
    cls = getattr(module, classname)
    return cls
