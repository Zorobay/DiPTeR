from src.gui.widgets.graphics.node import Node
from src.opengl.shader_types import INTERNAL_TYPE_FLOAT, INTERNAL_TYPE_RGB


class OutputNode(Node):

    def __init__(self, *args):
        super().__init__(*args, title="Output Node")

        #self.add_input(SOCKET_TYPE_RGB, "Shader")



