import inspect
import typing


def get_function_arguments(func: typing.Callable) -> typing.Mapping[str, inspect.Parameter]:
    return inspect.signature(func).parameters
