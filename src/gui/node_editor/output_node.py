from src.gui.node_editor.node import ShaderNode, Node
from src.gui.node_editor.node_scene import NodeScene
from src.gui.node_editor.socket import Socket
from src.opengl.internal_types import INTERNAL_TYPE_SHADER


class OutputNode(Node):

    def __init__(self, scene: NodeScene):
        super().__init__(scene, "Output Node", None)

        #self.add_input("Output Texture", None, INTER)
        self.input_socket = Socket(self)
        self._init_node()

    def _init_node(self):
        self.input_socket.edge_started.connect(self._spawn_edge)
        self._master_layout.addItem(self.input_socket, self._input_index, 0)




