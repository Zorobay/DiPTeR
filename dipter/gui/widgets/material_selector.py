from pathlib import Path

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QWidget, QPushButton, QComboBox, QHBoxLayout, QInputDialog, QFileDialog
from dipter.gui.node_editor.control_center import ControlCenter
from dipter.gui.node_editor.material import Material
from dipter.misc import material_serializer


class MaterialSelector(QWidget):
    material_changed = pyqtSignal(Material)

    def __init__(self, cc: ControlCenter, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cc = cc

        # Define components
        self._add_button = QPushButton("+")
        self._material_combo_box = QComboBox()
        self._save_material_button = QPushButton("Save Material")
        self._load_material_button = QPushButton("Load Material")
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

        self._save_material_button.clicked.connect(self._save_material)
        self._layout.addWidget(self._save_material_button)

        self._load_material_button.clicked.connect(self._load_material)
        self._layout.addWidget(self._load_material_button)

        self.setLayout(self._layout)

    def add_material(self, name: str):
        mat = self.cc.new_material(name)
        self._add_material(mat)

    def load_material(self, material_filepath: str):
        filename = Path(material_filepath).stem
        mat = self.cc.new_material(filename, material_filepath)
        self._add_material(mat)

    def _add_material(self, material: Material):
        self._material_ids.append(material.id())
        self._material_combo_box.addItem(material.name)
        self._material_combo_box.size()
        self._material_combo_box.setCurrentIndex(self._material_combo_box.count() - 1)

    def _add_button_clicked(self):
        name, ok = QInputDialog.getText(self, "New Material Name", "Material Name", text="Material{}".format(self.cc.get_num_materials()))

        if name and ok:
            self.add_material(name)

    def _selected_material_changed(self, index):
        self.cc.set_active_material_id(self._material_ids[index])

    def _save_material(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Material", filter="JSON (*.json)", directory="{}.json".format(
            self.cc.active_material.name))
        if filename:
            material_serializer.save_material(self.cc.active_material.get_material_output_node().get_backend_node(), filename)

    def _load_material(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Material", filter="Material (*.json)")

        if filename:
            self.load_material(filename)
