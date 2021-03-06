import json
import logging
import numbers
import typing
import uuid

import numpy as np
import torch

from dipter.misc import io as iofuncs
from dipter.misc import string_funcs
from dipter.node_graph.node import ShaderNode
from dipter.node_graph.node_socket import NodeSocket
from dipter.shaders.shaders.material_output_shader import MaterialOutputShader

SHADER = "shader"
POSITION = "position"
INPUTS = "inputs"
ARGUMENT = "arg"
VALUE = "value"
CONNECTED = "connected"
OUTPUT_SOCKET_INDEX = "output_socket_index"

_logger = logging.getLogger(__name__)


class Serializer(json.JSONEncoder):

    def default(self, obj):
        """Returns a JSON serializable form of a value."""
        try:
            return Serializer.serialize(obj)
        except ValueError:
            return super().default(obj)

    @classmethod
    def serialize(cls, obj):
        if isinstance(obj, uuid.UUID):
            return obj.hex
        elif isinstance(obj, torch.Tensor):
            return list(obj.cpu().numpy())
        elif isinstance(obj, np.ndarray):
            return list(obj)
        elif isinstance(obj, numbers.Number):
            return float(obj)
        else:
            raise ValueError()


def save_material(root_node: ShaderNode, filename: str):
    output_dict = dict()

    _save_material(root_node, output_dict)

    with open(filename, "w", encoding="utf-8") as f:
        json_str = json.dumps(output_dict, cls=Serializer, ensure_ascii=False)
        f.write(json_str)
        _logger.info("Saved material to file {}".format(filename))


def _save_material(node: ShaderNode, output_dict: dict):
    container = node.get_container()
    pos = None
    if container:
        pos = (container.x(), container.y())

    node_info = {
        SHADER: str(node.get_shader().__class__),
        POSITION: pos,
        INPUTS: []
    }

    for socket in node.get_input_sockets():
        arg = socket.label()
        if socket.is_connected():
            connected = True
            connected_node = socket.get_connected_nodes()
            connected_socket = socket.get_connected_sockets()
            assert len(connected_node) == 1 and len(connected_socket) == 1
            connected_node = connected_node[0]
            connected_socket = connected_socket[0]
            out_socket_index = connected_socket.get_index()
            value = dict()
            _save_material(connected_node, value)
        else:
            connected = False
            out_socket_index = -1
            value = socket.value()

        node_info[INPUTS].append({
            ARGUMENT: arg,
            CONNECTED: connected,
            OUTPUT_SOCKET_INDEX: out_socket_index,
            VALUE: value
        })

    output_dict[Serializer.serialize(node.id())] = node_info


def load_material(filepath, assign_numbers=True) -> typing.Tuple[ShaderNode, dict]:
    """
    Loads a saved material from file.
    :param filepath: The path to the material .json file.
    :param assign_numbers: if True will automatically assign unique node numbers to the loaded nodes.
    :return: the root node
    """
    taken_numbers = {}
    with open(filepath, "r", encoding="utf-8") as f:
        mat_dict = json.load(f)

    mon = ShaderNode(shader=MaterialOutputShader())
    _load_material(mat_dict, taken_numbers, assign_numbers, parent=mon)
    return mon, mat_dict


def _get_node_number(shader: str, taken_numbers: dict) -> int:
    if shader not in taken_numbers:
        taken_numbers[shader] = [None]

    numbers = taken_numbers[shader]

    for i, n in enumerate(numbers):
        if n is None:
            numbers[i] = i
            return i

    # If no empty slot is found, add the next bigger number to the end of the list
    num = len(numbers)
    numbers.append(num)
    return num


def _load_material(mat_dict: dict, taken_numbers: dict, assign_numbers: bool, parent: ShaderNode = None, parent_socket: NodeSocket = None,
                   output_index: int = None):
    for key_id in mat_dict:
        node_dict = mat_dict[key_id]
        shader = node_dict[SHADER]

        if "MaterialOutputShader" in shader:
            node = parent
        else:
            import_string = string_funcs.type_to_import_string(shader)
            cls = iofuncs.import_class_from_string(import_string)
            node = ShaderNode(shader=cls())
            if assign_numbers:
                node_num = _get_node_number(shader, taken_numbers)
                node.set_num(node_num)

            # Connect parents input socket to current node's output socket
            out_socket = node.get_output_socket(output_index)
            out_socket.connect_to(parent_socket)

        try:
            node._id = uuid.UUID(key_id)  # Explicitly set the uuid from file
        except ValueError as e:
            _logger.error("Failed to explicitly set the uuid of node {}. A new uuid will be randomized. Error: \n{}".format(node.label(), e))

        inputs = node_dict[INPUTS]
        for inp in inputs:
            arg = inp[ARGUMENT]
            connected = inp[CONNECTED]
            value = inp[VALUE]
            output_index = inp[OUTPUT_SOCKET_INDEX]
            socket = node.get_input_socket(arg)
            if connected:
                _load_material(value, taken_numbers, assign_numbers, parent=node, parent_socket=socket, output_index=output_index)
            else:
                if socket:
                    socket.set_value(torch.tensor(value, dtype=torch.float32))
                else:
                    raise RuntimeError("Can not find socket with label {} on node created from shader {} while loading material.".format(arg, shader))
