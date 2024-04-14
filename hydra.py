from PyQt6.QtWidgets import QTableWidgetItem, QComboBox, QDoubleSpinBox
from PyQt6.QtWidgets import QSpinBox, QMessageBox, QDialog, QVBoxLayout, QLabel
from PyQt6 import QtCore
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
import networkx as nx
import pygraphviz as pgv
from objects import objects_name_list, objects_dict


class ImageDialog(QDialog):
    def __init__(self, image_path):
        super().__init__()
        layout = QVBoxLayout()
        image_label = QLabel()
        pixmap = QPixmap(image_path)
        image_label.setPixmap(pixmap)
        layout.addWidget(image_label)
        self.setLayout(layout)
        self.setWindowTitle("Топология сети")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.exec()


class HydraTable():
    def AddHydraRow(self):
        self.HydraTableWidget.insertRow(self.HydraTableWidget.rowCount())
        self.HydraNumberSpinBox()
        self.HydraBeginningComboBox()
        self.HydraEndComboBox()

    def RemoveHydraRow(self):
        if self.HydraTableWidget.rowCount() == 1:
            QMessageBox().warning(
                None, "Единственная строка", "Вы не можете удалить единственную строку в таблице.")  # noqa E501
            return
        else:
            self.HydraTableWidget.removeRow(
                self.HydraTableWidget.rowCount()-1)

    def HydraNumberSpinBox(self):
        # Указываем столбец, для которого нужно установить SpinBox
        col = 0
        current_row = self.HydraTableWidget.rowCount() - 1
        # Получаем значение номера помещения из предыдущей строки
        prev_row = self.HydraTableWidget.rowCount() - 2
        prev_widget = self.HydraTableWidget.cellWidget(prev_row, col)
        prev_value = prev_widget.value() if prev_widget else 0
        sb = QSpinBox()
        sb.setMaximum(1000)
        sb.setValue(prev_value + 1)
        self.HydraTableWidget.setCellWidget(current_row, col, sb)

    def HydraBeginningComboBox(self):
        col = 1
        row = self.HydraTableWidget.rowCount() - 1
        cb = QComboBox()
        cb.addItems(objects_name_list)
        self.HydraTableWidget.setCellWidget(row, col, cb)

    def HydraEndComboBox(self):
        col = 2
        row = self.HydraTableWidget.rowCount() - 1
        cb = QComboBox()
        cb.addItems(objects_name_list)
        self.HydraTableWidget.setCellWidget(row, col, cb)

    def ChangeHydraComboBoxContents(self):
        for row in range(self.HydraTableWidget.rowCount()):
            self.HydraTableWidget.cellWidget(row, 1).clear()
            self.HydraTableWidget.cellWidget(row, 2).clear()
            self.HydraTableWidget.cellWidget(
                row, 1).addItems(objects_name_list)
            self.HydraTableWidget.cellWidget(
                row, 2).addItems(objects_name_list)

    def BuildTopology(self):
        graph = nx.MultiDiGraph()
        if len(objects_name_list) == 0:
            QMessageBox.warning(None, "Пустой список объектов",
                                "Список объектов пуст. Добавьте объекты на листе Объекты!")  # noqa E501
            return

        graph.add_nodes_from(objects_name_list)
        for row in range(self.HydraTableWidget.rowCount()):
            beginning_object_widget = self.HydraTableWidget.cellWidget(row, 1)
            end_object_widget = self.HydraTableWidget.cellWidget(row, 2)
            if beginning_object_widget is None or end_object_widget is None:
                QMessageBox.warning(
                    None, "Пустая ячейка", f"Строка {row + 1} содержит пустую ячейку. Пожалуйста, заполните все ячейки.")  # noqa E501
                return
            beginning_object = beginning_object_widget.currentText()
            end_object = end_object_widget.currentText()
            if not beginning_object or not end_object:
                QMessageBox.warning(
                    None, "Пустое значение", f"В строке {row + 1} есть пустое значение. Пожалуйста, заполните все значения.")  # noqa E501
                return
            graph.add_edge(beginning_object, end_object)

        if not self.IsGRPInGraph(graph):  # проверка наличия ГРП в графе
            QMessageBox.warning(
                None, "Отсутствие ГРП",
                "В графе отсутствуют объекты типа ГРП. Добавьте объекты ГРП на листе Объекты!")  # noqa E501
            return

        # проверка связи между ГРП и Потребитель
        if not self.IsConsumerConnected(graph):
            QMessageBox.warning(
                None, "Нет связи", "Нет связи между объектами типа ГРП и Потребитель!")  # noqa E501
            return

        if not self.IsGraphConnected(graph):  # проверка связности графа
            self.ShowTopologyPushButton.setEnabled(False)
            return

        self.ShowTopologyPushButton.setEnabled(True)
        # Визуализация графа
        A = nx.nx_agraph.to_agraph(graph)
        A.layout('dot')
        filename = f"graphs_pic/topology{graph.edges}.png"
        A.draw(filename)
        self.ShowTopology(filename)

    def ShowTopology(self, filename):
        image_path = filename
        ImageDialog(image_path)

    def IsGraphConnected(self, graph):
        is_connected = nx.is_weakly_connected(graph)
        if is_connected:
            return True
        if not is_connected:
            msgBox = QMessageBox()
            msgBox.setWindowTitle("Cвязность графа")
            msgBox.setText(
                "Граф не связан. \nВы уверены, что хотите продолжить?")
            msgBox.addButton(QMessageBox.StandardButton.Yes).setText("Да")
            msgBox.addButton(QMessageBox.StandardButton.No).setText("Нет")
            reply = msgBox.exec()
            if reply == QMessageBox.StandardButton.Yes:
                return True
            else:
                return False

    def IsGRPInGraph(self, graph):
        for node in graph.nodes():
            if objects_dict.get(node) == "ГРП":
                return True
        return False

    def IsConsumerConnected(self, graph):
        for beginning_object in objects_dict:
            if objects_dict[beginning_object] == "ГРП":
                visited = set()
                stack = [beginning_object]

                while stack:
                    current_object = stack.pop()
                    if current_object in visited:
                        continue
                    visited.add(current_object)

                    if objects_dict[current_object] == "Потребитель":
                        return True

                    for neighbor in graph.neighbors(current_object):
                        stack.append(neighbor)
        return False
