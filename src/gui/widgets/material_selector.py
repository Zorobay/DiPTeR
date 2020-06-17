from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QWidget, QPushButton, QComboBox, QHBoxLayout, QInputDialog

from src.gui.node_editor.control_center import ControlCenter
from src.gui.node_editor.material import Material


class MaterialSelector(QWidget):
    material_changed = pyqtSignal(Material)

    def __init__(self, cc: ControlCenter, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cc = cc

        # Define components
        self._add_button = QPushButton("+")
        self._material_combo_box = QComboBox()
        self._layout = QHBoxLayout()

        # Define data
        self._material_ids = []
        self._active_material = None

        self._init_widget()

    def _init_widget(self):
        self._layout.setAlignment(Qt.AlignLeft)

        self._add_button.clicked.connect(self._add_button_clicked)
        self._add_button.setFixedWidth(20)
        self._layout.addWidget(self._add_button)

        self._material_combo_box.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._material_combo_box.currentIndexChanged.connect(self._selected_material_changed)
        self._layout.addWidget(self._material_combo_box)

        self.setLayout(self._layout)

    def add_material(self, name: str):
        mat = self.cc.new_material(name)
        self._material_ids.append(mat.id())
        self._material_combo_box.addItem(name)
        self._material_combo_box.size()
        self._material_combo_box.setCurrentIndex(self._material_combo_box.count()-1)

        # Add a default material output shader to the material
        #mat.add_node(MaterialOutputShader)

    def _add_button_clicked(self):
        name, ok = QInputDialog.getText(self, "New Material Name", "Material Name", text="Material{}".format(self.cc.get_num_materials()))

        if name and ok:
            self.add_material(name)

    def _selected_material_changed(self, index):
        self.cc.set_active_material_id(self._material_ids[index])
