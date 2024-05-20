from PyQt6.QtWidgets import QTableWidgetItem, QPushButton

import os
import json


class ResultTable():
    def LoadVariantDataForRow(self, variant_name):
        if os.path.exists("variant_data.json"):
            with open("variant_data.json", "r", encoding='utf-8') as json_file:
                variant_data = json.load(json_file)
        else:
            print("Нет данных")
            return
        if variant_name in variant_data:
            data = variant_data[variant_name]
            self.ClearHydraTable()
            self.ClearObjectsTable()
            objects_data = data.get("Объекты", {})
            self.LoadTableData(self.ObjectsTableWidget, objects_data)
            self.ChangeHydraComboBoxContents()
            hydra_data = data.get("Гидравлика", {})
            self.LoadTableData(self.HydraTableWidget, hydra_data)
        else:
            print(f"Нет данных для варианта {variant_name}")

    def VariantExistsInResultTable(self, variant_name):
        for row in range(self.ResultTableWidget.rowCount()):
            if self.ResultTableWidget.item(row, 0).text() == variant_name:
                return True
        return False

    def AddResultRow(self):
        for i in range(self.VariantComboBox.count()):
            variant_name = self.VariantComboBox.itemText(i)
            if self.VariantExistsInResultTable(variant_name):
                continue
            self.LoadVariantDataForRow(variant_name)
            self.ResultTableWidget.insertRow(self.ResultTableWidget.rowCount())
            self.ResultTableWidget.setItem(
                self.ResultTableWidget.rowCount() - 1, 0,
                QTableWidgetItem(variant_name))
            self.ResultGetMinPressure()
            self.ResultGetMaxVelocity()
            self.ResultTopologyButton()
            self.ResultPiezoButton()

    def ResultGetMinPressure(self):
        col = 1
        row = self.ResultTableWidget.rowCount() - 1
        min_pressure = float(self.ObjectsTableWidget.item(0, 4).text())
        min_object = self.ObjectsTableWidget.item(0, 2).text()
        for obj_row in range(self.ObjectsTableWidget.rowCount()):
            pressure = float(self.ObjectsTableWidget.item(obj_row, 4).text())
            object_ = self.ObjectsTableWidget.item(obj_row, 2).text()
            if pressure < min_pressure:
                min_pressure = pressure
                min_object = object_
        item = QTableWidgetItem(f"{min_object}: {min_pressure} Па")
        self.ResultTableWidget.setItem(row, col, item)

    def ResultGetMaxVelocity(self):
        col = 2
        row = self.ResultTableWidget.rowCount() - 1
        max_velocity = float(self.HydraTableWidget.item(0, 7).text())
        max_vel_path = f"{self.HydraTableWidget.cellWidget(0, 1).currentText()}->{self.HydraTableWidget.cellWidget(0, 2).currentText()}"  # noqa E501
        for hydra_row in range(self.HydraTableWidget.rowCount()):
            velocity = float(self.HydraTableWidget.item(hydra_row, 7).text())
            vel_path = f"{self.HydraTableWidget.cellWidget(hydra_row, 1).currentText()}->{self.HydraTableWidget.cellWidget(hydra_row, 2).currentText()}"  # noqa E501
            if velocity > max_velocity:
                max_velocity = velocity
                max_vel_path = vel_path
        item = QTableWidgetItem(f"{max_vel_path}: {max_velocity} м/с")
        self.ResultTableWidget.setItem(row, col, item)

    def ResultTopologyButton(self):
        col = 3
        row = self.ResultTableWidget.rowCount() - 1
        variant_name = self.ResultTableWidget.item(row, 0).text()
        topology_button = QPushButton("Показать Топологию")
        topology_button.clicked.connect(
            lambda: self.ShowTopology(
                f"graphs_pic/topology_{variant_name}.png"))
        self.ResultTableWidget.setCellWidget(row, col, topology_button)

    def ResultPiezoButton(self):
        col = 4
        row = self.ResultTableWidget.rowCount() - 1
        variant_name = self.ResultTableWidget.item(row, 0).text()
        piezo_button = QPushButton("Показать пьезометр")
        piezo_button.clicked.connect(
            lambda: self.ShowTopology(f"piezo_pic/{variant_name}.png"))
        self.ResultTableWidget.setCellWidget(row, col, piezo_button)
