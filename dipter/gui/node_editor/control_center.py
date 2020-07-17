import logging
import typing
import uuid

from PyQt5.QtCore import QObject, pyqtSignal

from dipter.gui.node_editor.g_edge import GEdge
from dipter.gui.node_editor.material import Material
from dipter.gui.node_editor.node_scene import NodeScene
from dipter.shaders.material_output_shader import MaterialOutputShader
from dipter.shaders.shader_super import FunctionShader

_logger = logging.getLogger(__name__)


class ControlCenter(QObject):
    active_material_changed = pyqtSignal(Material)
    edge_spawned = pyqtSignal(GEdge)

    def __init__(self):
        super().__init__()

        # Define data
        self._materials = {}
        self._active_material = None

        self._init_control_center()

    def _init_control_center(self):
        pass

    @property
    def active_material(self) -> typing.Union[Material, None]:
        """
        Reference to the active material or None if there is no active material.
        """
        return self._active_material

    @property
    def active_scene(self) -> typing.Union[NodeScene, None]:
        """
        Reference to the active scene contained in the active material. This value if None if there is no material is active.
        """
        if self.active_material:
            return self.active_material.node_scene

        return None

    def get_num_materials(self) -> int:
        """
        Get the number of materials.
        :return: the number of materials
        """
        return len(self._materials)

    def new_material(self, name: str, filename: str = None) -> typing.Union[Material, None]:
        """
        Adds a new Material to the Node Editor and returns it. The title must be unique.
        :param name: The display title of the Material to be created.
        :param filename: (Optional) a path to a material save file to load from.
        :return: the newly created Material or None is the title was not unique.
        """
        for _, m in self._materials.items():
            if m.name == name:
                newname = name.split(".")
                if len(newname > 1):
                    newname = newname[-2]

                last_char = newname[-1]
                try:
                    last_digit = int(last_char)
                    last_digit += 1
                    newname[-1] = str(last_digit)
                except ValueError:
                    newname += "1"
                    pass
                _logger.warning("Material with title {} already exists! Renaming new material to {}.".format(m.name, newname))
                name = newname

        material = Material(self, name, material_filepath=filename)
        self._materials[material.id()] = material
        _logger.debug("New material {} ({}) added.".format(material.name, material.id()))

        return material

    def set_active_material_id(self, material_id: uuid.UUID) -> bool:
        """
        Set the material with id 'material_id' to the new active material. If the id is invalid, will return False, else True.
        :param material_id: the id of the material to set as active.
        """
        for id_, m in self._materials.items():
            if id_ == material_id:
                self._active_material = m
                self.active_material_changed.emit(m)
                _logger.debug("Set new active material {} ({}).".format(m.name, m.id()))
                return True

        _logger.error("Id <{}> does not match any material. New active material not set.".format(material_id))
        return False

    def new_node(self, shader: typing.Type[FunctionShader]):
        """
        Create a new node for the active material. Returns False if there is no active material, else returns True.

        :param shader:
        :return:
        """
        if self.active_material:
            self.active_material.add_node(shader)
            return True

        return False

    def delete_node(self, node_id: uuid):
        """
        Deletes a node from the active material.

        :param node_id: The UUID of the node to be deleted
        :return: True if successful, False otherwise.
        """
        if self.active_material:
            return self.active_material.delete_node(node_id)

        return False
