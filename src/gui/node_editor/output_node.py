from src.gui.node_editor.node import Node, NodeGraphics
from src.gui.node_editor.node_scene import NodeScene
from src.gui.node_editor.socket import Socket
from src.opengl.shader_types import INTERNAL_TYPE_ARRAY_RGBA


class OutputNode(NodeGraphics):

    def __init__(self, scene: NodeScene):
        super().__init__(scene, "Output Node", None)

        self.input_socket = Socket()
        self._init_node()

    def _init_node(self):
        self.input_socket.edge_started.connect(lambda s: self.edge_started.emit(self.id, s))
        self._master_layout.addItem(self.input_socket, self._input_index, 0)




