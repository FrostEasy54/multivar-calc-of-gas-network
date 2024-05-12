from PyQt6.QtWidgets import QTableWidgetItem, QComboBox, QDoubleSpinBox
from PyQt6.QtWidgets import QSpinBox, QMessageBox, QDialog, QVBoxLayout, QLabel
from PyQt6 import QtCore
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

from prettytable import PrettyTable
import numpy as np
import networkx as nx
import pygraphviz as pgv

import math
from objects import objects_name_list, objects_dict
from pipes import pipe_type_dict, roughness_factor_dict

NATURAL_GAS_VISCOSITY = 0.0000143
GAS_DENSITY = 0.73
edges_vector = []
gas_velocity_vector = []
pipe_diameter_vector = []
pipe_length_vector = []
Reynolds_number_vector = []
roughness_factor_vector = []
Darsi_friction_factor_vector = []
hydraulic_friction_factor = []
R_factor_vector = []
incidence_matrix_tr = 0
R_matrix = 0


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
        self.HydraLengthSpinBox()
        self.HydraPathConsumptionDoubleSpinBox()
        self.HydraPipeTypeComboBox()
        self.HydraPipeDiameter()
        self.HydraGasVelocity()

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

    def HydraLengthSpinBox(self):
        col = 3
        row = self.HydraTableWidget.rowCount() - 1
        sb = QSpinBox()
        sb.setMaximum(1000)
        sb.setMinimum(0)
        self.HydraTableWidget.setCellWidget(row, col, sb)

    def HydraPathConsumptionDoubleSpinBox(self):
        col = 4
        row = self.HydraTableWidget.rowCount() - 1
        sb = QDoubleSpinBox()
        sb.setMaximum(1000)
        sb.setMinimum(0)
        self.HydraTableWidget.setCellWidget(row, col, sb)
        self.HydraTableWidget.cellWidget(
            row, col).valueChanged.connect(self.HydraGasVelocity)

    def HydraPipeTypeComboBox(self):
        col = 5
        row = self.HydraTableWidget.rowCount() - 1
        cb = QComboBox()
        cb.addItems(pipe_type_dict.keys())
        self.HydraTableWidget.setCellWidget(row, col, cb)
        self.HydraTableWidget.cellWidget(
            row, col).currentTextChanged.connect(self.HydraPipeDiameter)
        self.HydraTableWidget.cellWidget(
            row, col).currentTextChanged.connect(self.HydraGasVelocity)

    def HydraPipeDiameter(self):
        col = 6
        row = self.HydraTableWidget.rowCount() - 1
        combo_box_item = self.HydraTableWidget.cellWidget(row, 5)
        selected_pipe = combo_box_item.currentText()
        diameter = pipe_type_dict.get(selected_pipe, "Нет данных")
        diameter_item = QTableWidgetItem(str(diameter))
        self.HydraTableWidget.setItem(row, col, diameter_item)

    def HydraGasVelocity(self):
        col = 7
        row = self.HydraTableWidget.rowCount() - 1
        diameter = float(self.HydraTableWidget.item(row, 6).text())
        Q = float(self.HydraTableWidget.cellWidget(row, 4).value())
        area = (0.25 * math.pi * (diameter / 1000) ** 2)
        velocity = Q / (area * 3600)
        velocity_item = QTableWidgetItem(str(velocity))
        self.HydraTableWidget.setItem(row, col, velocity_item)

    def CreateEdgesArray(self):
        edges_vector.clear()
        for row in range(self.HydraTableWidget.rowCount()):
            beginning_edge_text = str(
                self.HydraTableWidget.cellWidget(row, 1).currentText())
            end_edge_text = str(
                self.HydraTableWidget.cellWidget(row, 2).currentText())
            edge = tuple((beginning_edge_text, end_edge_text))
            edges_vector.append(edge)
        print('Вектор ребер')
        print(edges_vector)

    def CreateVelocityArray(self):
        gas_velocity_vector.clear()
        for row in range(self.HydraTableWidget.rowCount()):
            velocity = float(self.HydraTableWidget.item(row, 7).text())
            gas_velocity_vector.append(velocity)
        print(gas_velocity_vector)

    def CreateDiameterArray(self):
        pipe_diameter_vector.clear()
        for row in range(self.HydraTableWidget.rowCount()):
            diameter = float(self.HydraTableWidget.item(row, 6).text())
            pipe_diameter_vector.append(diameter)
        print(pipe_diameter_vector)

    def CreateLengthArray(self):
        pipe_length_vector.clear()
        for row in range(self.HydraTableWidget.rowCount()):
            length = int(self.HydraTableWidget.cellWidget(row, 3).value())
            pipe_length_vector.append(length)
        print(pipe_length_vector)

    def CalculateReynoldsNumber(self):
        Reynolds_number_vector.clear()
        for i in range(len(gas_velocity_vector)):
            Reynolds_number = gas_velocity_vector[i] * \
                pipe_diameter_vector[i] / 1000 / NATURAL_GAS_VISCOSITY
            Reynolds_number_vector.append(Reynolds_number)
        print(Reynolds_number_vector)

    def CreatePipeRoughnessFactor(self):
        roughness_factor_vector.clear()
        for i in range(len(gas_velocity_vector)):
            material_name = self.HydraTableWidget.cellWidget(
                i, 5).currentText().split()[0]
            if material_name in roughness_factor_dict:
                roughness_factor_vector.append(
                    roughness_factor_dict[material_name])
            else:
                roughness_factor_vector.append(0)
        print("---------------------")
        print(roughness_factor_vector)

    def CalculateDarsiFrictionFactor(self):
        Darsi_friction_factor_vector.clear()
        for i in range(len(gas_velocity_vector)):
            roughness_factor = roughness_factor_dict.get('Сталь')
            Reynolds_roughness_diameter_ratio = Reynolds_number_vector[i] * \
                roughness_factor / pipe_diameter_vector[i]
            if Reynolds_number_vector[i] == 0:
                friction_factor = 1 / \
                    ((2 * math.log10(3.7 /
                     (roughness_factor/pipe_diameter_vector[i])))**2)
                Darsi_friction_factor_vector.append(friction_factor)
            elif Reynolds_number_vector[i] < 2300:
                friction_factor = 64 / Reynolds_number_vector[i]
                Darsi_friction_factor_vector.append(friction_factor)
            elif Reynolds_roughness_diameter_ratio < 23 and \
                    Reynolds_number_vector[i] < 125000:
                friction_factor = 0.3164 / (Reynolds_number_vector[i] ** 0.25)
                Darsi_friction_factor_vector.append(friction_factor)
            elif Reynolds_roughness_diameter_ratio < 23 and \
                    Reynolds_number_vector[i] > 125000:
                friction_factor = 0.0032 + \
                    (0.221 / (Reynolds_number_vector[i] ** 0.237))
                Darsi_friction_factor_vector.append(friction_factor)
            elif Reynolds_roughness_diameter_ratio >= 23 and\
                    Reynolds_roughness_diameter_ratio < 560:
                friction_factor = 0.11 * \
                    (((roughness_factor /
                     pipe_diameter_vector[i]) + (
                        68/Reynolds_number_vector[i])) ** 0.25)
                Darsi_friction_factor_vector.append(friction_factor)
            elif Reynolds_roughness_diameter_ratio >= 560:
                friction_factor = 1 / \
                    ((2 * math.log10(3.7 /
                     (roughness_factor/pipe_diameter_vector[i])))**2)
                Darsi_friction_factor_vector.append(friction_factor)
        print(Darsi_friction_factor_vector)

    def CalculateHydraulicFrictionFactor(self):
        hydraulic_friction_factor.clear()
        for i in range(len(gas_velocity_vector)):
            friction_factor = (GAS_DENSITY / 2) * \
                Darsi_friction_factor_vector[i] * pipe_length_vector[i] / (
                    pipe_diameter_vector[i]/1000) * 16/(
                        (3600*math.pi*((pipe_diameter_vector[i]/1000)**2))**2)
            hydraulic_friction_factor.append(friction_factor)
        print("vector S\n")
        print(hydraulic_friction_factor)

    def CalculateRFactor(self):
        R_factor_vector.clear()
        for i in range(len(gas_velocity_vector)):
            R_factor = 1 / ((hydraulic_friction_factor[i]) ** 0.5)
            R_factor_vector.append(R_factor)
        print("Vector R\n")
        print(R_factor_vector)

    def CreateIncidenceMatrix(self):
        incidence_matrix = np.zeros(
            (len(objects_name_list), len(edges_vector)), dtype=int)
        edge_indices = {edge: i for i, edge in enumerate(edges_vector)}
        node_indices = {node: i for i, node in enumerate(objects_name_list)}
        for edge in edges_vector:
            node1, node2 = edge
            edge_index = edge_indices[edge]
            node1_index = node_indices[node1]
            node2_index = node_indices[node2]
            incidence_matrix[node1_index, edge_index] = 1
            incidence_matrix[node2_index, edge_index] = -1
        incidence_matrix_tr = incidence_matrix.transpose()

        table = PrettyTable()
        table.field_names = ['Участки/Узлы'] + objects_name_list
        for i, row in enumerate(incidence_matrix_tr):
            table.add_row([f"{edges_vector[i]}"] + list(row))

        # Выводим таблицу с использованием PrettyTable
        print(table)

    def CreateRFactorMatrix(self):
        matrix_size = len(R_factor_vector)
        diagonal_R_matrix = np.zeros((matrix_size, matrix_size))
        np.fill_diagonal(diagonal_R_matrix, R_factor_vector)
        R_matrix = diagonal_R_matrix
        print(R_matrix)

    def CalculateM0Matrix(self):
        if incidence_matrix_tr == 0:
            QMessageBox().warning(
                None, "Матрица не существует",
                "Матрицы инцидентности не сущесвует.")
            return
        incidence_matrix = incidence_matrix_tr.transpose()
        M0_matrix = np.matmul(
            np.matmul(incidence_matrix, R_matrix), incidence_matrix_tr)
        return M0_matrix

    def CalculateAll(self):
        self.CreateEdgesArray()

        self.CreateVelocityArray()
        self.CreateDiameterArray()
        self.CreateLengthArray()
        self.CalculateReynoldsNumber()
        self.CreatePipeRoughnessFactor()
        self.CalculateDarsiFrictionFactor()
        self.CalculateHydraulicFrictionFactor()
        self.CalculateRFactor()

        self.CreateIncidenceMatrix()  # нужно чтоб существовал Edges Array
        self.CreateRFactorMatrix()  # нужно чтоб существовал R Factor
        self.CalculateM0Matrix()  # нужно чтоб существовала матрица инцидентности и матрица R

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
