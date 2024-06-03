from PyQt6.QtWidgets import QTableWidgetItem, QComboBox
from PyQt6.QtWidgets import QSpinBox, QMessageBox, QFileDialog

import csv

objects_name_list = []
objects_dict = {}


class ObjectsTable():
    def AddObjectsRow(self):
        ''' object_name_item = self.ObjectsTableWidget.item(
            self.ObjectsTableWidget.rowCount() - 1, 2)
        if object_name_item is None or object_name_item.text().strip() == "":
            QMessageBox().warning(
                None, "Пустое поле", "Пожалуйста, введите условное обозначение объекта.")  # noqa E501
            return'''
        self.ObjectsTableWidget.insertRow(self.ObjectsTableWidget.rowCount())
        self.ObjectsTypeComboBox()
        self.ObjectsNumberSpinBox()

    def RemoveObjectsRow(self):
        if self.ObjectsTableWidget.rowCount() == 1:
            QMessageBox().warning(
                None, "Единственная строка", "Вы не можете удалить единственную строку в таблице.")  # noqa E501
            return
        else:
            self.ObjectsTableWidget.removeRow(
                self.ObjectsTableWidget.rowCount()-1)
        self.UpdateObjects()

    def ObjectsNumberSpinBox(self):
        col = 0
        current_row = self.ObjectsTableWidget.rowCount() - 1
        prev_row = self.ObjectsTableWidget.rowCount() - 2
        prev_widget = self.ObjectsTableWidget.cellWidget(prev_row, col)
        prev_value = prev_widget.value() if prev_widget else 0
        sb = QSpinBox()
        sb.setMaximum(1000)
        sb.setValue(prev_value + 1)
        self.ObjectsTableWidget.setCellWidget(current_row, col, sb)

    def ObjectsTypeComboBox(self):
        col = 1
        row = self.ObjectsTableWidget.rowCount() - 1
        objects_type = ["Точка", "Потребитель", "Источник"]
        cb = QComboBox()
        cb.addItems(objects_type)
        self.ObjectsTableWidget.setCellWidget(row, col, cb)

    def UpdateObjects(self):
        objects_name_list.clear()
        objects_dict.clear()
        for row in range(self.ObjectsTableWidget.rowCount()):
            object_name_item = self.ObjectsTableWidget.item(row, 2)
            object_type_cb = self.ObjectsTableWidget.cellWidget(row, 1)
            if object_name_item is not None and object_type_cb is not None:
                object_name = object_name_item.text()
                object_type = object_type_cb.currentText()
                if object_name not in objects_name_list:
                    objects_name_list.append(object_name)
                    objects_dict[object_name] = object_type
        print(objects_dict)

    def ClearObjectsTable(self):
        self.ObjectsTableWidget.setRowCount(0)
        self.AddObjectsRow()
        objects_name_list.clear()
        objects_dict.clear()

    def ObjectsSaveToCSV(self):
        try:
            file_dialog = QFileDialog()
            path, _ = file_dialog.getSaveFileName(
                None, "Сохранить Объекты", "",
                "CSV Files (*.csv);;All Files (*)")
            if path:
                with open(path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(
                        ['№', 'Тип объекта', 'Условное обозначение'])
                    for row in range(self.ObjectsTableWidget.rowCount()):
                        object_number = self.ObjectsTableWidget.cellWidget(
                            row, 0).value()
                        object_type = self.ObjectsTableWidget.cellWidget(
                            row, 1).currentText()
                        object_name = self.ObjectsTableWidget.item(
                            row, 2).text()
                        writer.writerow(
                            [object_number, object_type, object_name])
                QMessageBox().information(None, "Сохранено",
                                          f"Данные успешно сохранены в файл: {path}")  # noqa E501
        except Exception as e:
            QMessageBox().critical(None, "Ошибка",
                                   f"Произошла ошибка при сохранении: {e}")

    def ObjectsLoadFromCSV(self):
        try:
            file_dialog = QFileDialog()
            path, _ = file_dialog.getOpenFileName(
                None, "Загрузить Объекты",
                "", "CSV Files (*.csv)")
            if path:
                with open(path, 'r', encoding='utf-8') as csvfile:
                    reader = csv.reader(csvfile)
                    self.ClearObjectsTable()
                    for row, row_data in enumerate(reader):
                        if row > 0:
                            self.AddObjectsRow()
                            for col, data in enumerate(row_data):
                                if col == 0:
                                    self.ObjectsTableWidget.cellWidget(
                                        row - 1, col).setValue(int(data))
                                elif col == 1:
                                    self.ObjectsTableWidget.cellWidget(
                                        row - 1, col).setCurrentText(str(data))
                                else:
                                    item = QTableWidgetItem(data)
                                    self.ObjectsTableWidget.setItem(
                                        row - 1, col, item)
            self.RemoveObjectsRow()
            self.ChangeHydraComboBoxContents()
            QMessageBox().information(None, "Импорт завершен",
                                      "Данные успешно импортированы.")
        except Exception as e:
            QMessageBox().critical(None, "Ошибка",
                                   f"Произошла ошибка при загрузке: {e}")
