from PyQt6.QtWidgets import QTableWidgetItem, QComboBox, QDoubleSpinBox
from PyQt6.QtWidgets import QSpinBox, QMessageBox
from PyQt6 import QtCore

objects_name_list = []
objects_dict = {}


class ObjectsTable():
    def AddObjectsRow(self):
        object_name_item = self.ObjectsTableWidget.item(
            self.ObjectsTableWidget.rowCount() - 1, 2)
        if object_name_item is None or object_name_item.text().strip() == "":
            message_box = QMessageBox()
            message_box.warning(
                None, "Пустое поле", "Пожалуйста, введите условное обозначение объекта.")  # noqa E501
            message_box.setFixedSize(500, 200)
            return
        self.ObjectsTableWidget.insertRow(self.ObjectsTableWidget.rowCount())
        self.ObjectsTypeComboBox()
        self.ObjectsNumberSpinBox()

    def RemoveObjectsRow(self):
        if self.ObjectsTableWidget.rowCount() == 1:
            message_box = QMessageBox()
            message_box.warning(
                None, "Единственная строка", "Вы не можете удалить единственную строку в таблице.")  # noqa E501
            message_box.setFixedSize(500, 200)
            return
        else:
            self.ObjectsTableWidget.removeRow(
                self.ObjectsTableWidget.rowCount()-1)
        self.UpdateObjects()

    def ObjectsNumberSpinBox(self):
        # Указываем столбец, для которого нужно установить SpinBox
        col = 0
        current_row = self.ObjectsTableWidget.rowCount() - 1
        # Получаем значение номера помещения из предыдущей строки
        prev_row = self.ObjectsTableWidget.rowCount() - 2
        prev_widget = self.ObjectsTableWidget.cellWidget(prev_row, col)
        prev_value = prev_widget.value() if prev_widget else 0
        sb = QSpinBox()
        sb.setMaximum(1000)
        sb.setValue(prev_value + 1)
        self.ObjectsTableWidget.setCellWidget(current_row, col, sb)

    def ObjectsTypeComboBox(self):
        # Указываем столбец, для которого нужно установить combobox
        col = 1
        row = self.ObjectsTableWidget.rowCount() - 1
        # Устанавливаем combobox для каждой ячейки в втором столбце
        objects_type = ["Точка", "Потребитель", "ГРП"]
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
