from PyQt6.QtWidgets import QTableWidgetItem, QPushButton


class ResultTable():
    def AddResultRow(self):
        for i in range(self.VariantComboBox.count()):
            item_name = self.VariantComboBox.itemText(i)
            self.ResultTableWidget.insertRow(self.ResultTableWidget.rowCount())
            self.ResultTableWidget.setItem(
                self.ResultTableWidget.rowCount() - 1, 0,
                QTableWidgetItem(item_name))
            self.ResultTopologyButton()
            self.ResultPiezoButton()

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
