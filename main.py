import os
import sys
from PyQt6.QtWidgets import QMainWindow, QHeaderView, QApplication
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from PyQt6 import uic
from objects import ObjectsTable
from hydra import HydraTable
ui_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'gasgui.ui'))


class MyGUI(QMainWindow, ObjectsTable, HydraTable):

    def __init__(self):
        super(MyGUI, self).__init__()
        uic.loadUi(ui_path, self)
        self.setWindowTitle(
            "Многовариантый гидравлический расчет сетей низкого газоснабжения")
        self.ObjectsTableInit()
        self.HydraTableInit()

    # Инициализация таблицы Объекты, все методы с ней писать сюда!
    def ObjectsTableInit(self):
        self.ObjectsTableWidget.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        # соединение кнопок для таблицы Объекты
        self.AddRowObjectsPushButton.clicked.connect(self.AddObjectsRow)
        self.RemoveRowObjectsPushButton.clicked.connect(self.RemoveObjectsRow)
        self.ExportObjectsOnHydraPushButton.clicked.connect(
            self.ChangeHydraComboBoxContents)

        self.ActionSaveObjects.triggered.connect(self.ObjectsSaveToCSV)
        self.ActionLoadObjects.triggered.connect(self.ObjectsLoadFromCSV)
        # заполнение первой строки
        self.ObjectsTypeComboBox()
        self.ObjectsNumberSpinBox()

        self.ObjectsTableWidget.cellChanged.connect(self.UpdateObjects)

    # Инициализация таблицы Гидравлика, все методы с ней писать сюда!
    def HydraTableInit(self):
        self.HydraTableWidget.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        # соединение кнопок для таблицы Гидравлика
        self.AddRowHydraPushButton.clicked.connect(self.AddHydraRow)
        self.RemoveRowHydraPushButton.clicked.connect(self.RemoveHydraRow)
        self.ShowTopologyPushButton.clicked.connect(self.ShowTopology)
        self.BuildTopologyPushButton.clicked.connect(self.BuildTopology)
        self.CalculatePushButton.clicked.connect(self.CalculateAll)

        self.ActionSaveHydra.triggered.connect(self.HydraSaveToCSV)
        self.ActionLoadHydra.triggered.connect(self.HydraLoadFromCSV)
        # заполнение первой строки
        self.HydraBeginningComboBox()
        self.HydraEndComboBox()
        self.HydraNumberSpinBox()
        self.HydraLengthSpinBox()
        self.HydraPathConsumptionDoubleSpinBox()
        self.HydraPipeTypeComboBox()
        self.HydraPipeDiameter()
        self.HydraGasVelocity()


def main():
    app = QApplication(sys.argv)
    window = MyGUI()
    window.show()
    app.exec()


if __name__ == '__main__':
    main()
