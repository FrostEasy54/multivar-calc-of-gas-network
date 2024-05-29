import os
import sys
from PyQt6.QtWidgets import QMainWindow, QHeaderView, QApplication
from PyQt6 import uic
from objects import ObjectsTable
from hydra import HydraTable
from results import ResultTable
ui_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'gasgui.ui'))


class MyGUI(QMainWindow, ObjectsTable, HydraTable, ResultTable):

    def __init__(self):
        super(MyGUI, self).__init__()
        uic.loadUi(ui_path, self)
        self.setWindowTitle(
            "Многовариантый гидравлический расчет сетей низкого газоснабжения")
        self.ObjectsTableInit()
        self.HydraTableInit()
        self.ResultTableInit()

    # Инициализация таблицы Объекты
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
        self.ActionObjectsPiezo.triggered.connect(self.PlotPiezo)
        self.actionClearObjects.triggered.connect(self.ClearObjectsTable)
        # заполнение первой строки
        self.ObjectsTypeComboBox()
        self.ObjectsNumberSpinBox()

        self.ObjectsTableWidget.cellChanged.connect(self.UpdateObjects)

    # Инициализация таблицы Гидравлика
    def HydraTableInit(self):
        self.HydraTableWidget.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        # соединение кнопок для таблицы Гидравлика
        self.AddRowHydraPushButton.clicked.connect(self.AddHydraRow)
        self.RemoveRowHydraPushButton.clicked.connect(self.RemoveHydraRow)
        self.BuildTopologyPushButton.clicked.connect(self.BuildTopology)
        self.CalculatePushButton.clicked.connect(self.CalculateAll)
        self.AddVariantPushButton.clicked.connect(self.HydraAddVariant)
        self.DeleteVariantPushButton.clicked.connect(self.HydraDropVariant)
        self.SaveVariantPushButton.clicked.connect(self.SaveVariantData)
        self.VariantComboBox.currentTextChanged.connect(self.LoadVariantData)

        self.ActionSaveHydra.triggered.connect(self.HydraSaveToCSV)
        self.ActionLoadHydra.triggered.connect(self.HydraLoadFromCSV)
        self.actionClearHydra.triggered.connect(self.ClearHydraTable)
        # заполнение первой строки
        self.HydraBeginningComboBox(0)
        self.HydraEndComboBox(0)
        self.HydraNumberSpinBox(0)
        self.HydraLengthSpinBox(0)
        self.HydraPathConsumptionDoubleSpinBox(0)
        self.HydraPipeTypeComboBox(0)
        self.HydraPipeDiameter(0)

    def ResultTableInit(self):
        self.ResultTableWidget.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.GetResultPushButton.clicked.connect(self.AddResultRow)


def main():
    app = QApplication(sys.argv)
    window = MyGUI()
    window.show()
    app.exec()


if __name__ == '__main__':
    main()
