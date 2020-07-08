import json
import logging
import numbers
import uuid
import numpy as np
import torch
from misc import string_funcs
from node_graph.node_socket import NodeSocket
from src.misc import io as iofuncs
from src.node_graph.node import ShaderNode
from src.shaders.material_output_shader import MaterialOutputShader

SHADER = "shader"
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
            return list(obj.numpy())
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
    node_info = {
        SHADER: str(node.get_shader().__class__),
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


def load_material(filename) -> ShaderNode:
    """Loads a saved material from file."""
    with open(filename, "r", encoding="utf-8") as f:
        mat_dict = json.load(f)

    mon = ShaderNode(shader=MaterialOutputShader())
    _load_material(mat_dict, parent=mon)
    return mon


def _load_material(mat_dict: dict, parent: ShaderNode = None, parent_socket: NodeSocket = None, output_index: int = None):
    for key_id in mat_dict:
        node_dict = mat_dict[key_id]
        shader = node_dict[SHADER]
        if "MaterialOutputShader" in shader:
            node = parent
        else:
            importstring = string_funcs.type_to_import_string(shader)
            cls = iofuncs.import_class_from_string(importstring)
            node = ShaderNode(shader=cls())

            # Connect parents input socket to current node's output socket
            out_socket = node.get_output_socket(output_index)
            out_socket.connect_to(parent_socket)

        inputs = node_dict[INPUTS]
        for inp in inputs:
            arg = inp[ARGUMENT]
            connected = inp[CONNECTED]
            value = inp[VALUE]
            output_index = inp[OUTPUT_SOCKET_INDEX]
            socket = node.get_input_socket(arg)
            if connected:
                _load_material(value, parent=node, parent_socket=socket, output_index=output_index)
            else:
                if socket:
                    socket.set_value(torch.tensor(value, dtype=torch.float32))
                else:
                    raise RuntimeError("Can not find socket with label {} on node created from shader {} while loading material.".format(arg, shader))


if __name__ == '__main__':
    node = load_material("cloud material.json")
    print(node)
    print(node.get_ancestor_nodes())
