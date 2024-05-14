from PyQt6.QtWidgets import QTableWidgetItem, QComboBox
from PyQt6.QtWidgets import QDoubleSpinBox, QFileDialog
from PyQt6.QtWidgets import QSpinBox, QMessageBox, QDialog, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

from prettytable import PrettyTable
import numpy as np
import networkx as nx

import math
import csv

from objects import objects_name_list, objects_dict
from pipes import pipe_type_dict, roughness_factor_dict

NATURAL_GAS_VISCOSITY = 0.0000143
GAS_DENSITY = 0.73
edges_vector = []
path_consumption_vector = []
node_consumption_vector = []
gas_velocity_vector = []
pipe_diameter_vector = []
pipe_length_vector = []
Reynolds_number_vector = []
roughness_factor_vector = []
Darsi_friction_factor_vector = []
hydraulic_friction_factor = []
R_factor_vector = []
m0_mult_press_vector_plus_Q = []
x_k_vector = []
s1_x_k_vector = []
sigma_g_k_vector = []
delta_p_k_plus_1_vector = []


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

    def CreatePathConsumptionArray(self):
        path_consumption_vector.clear()
        for row in range(self.HydraTableWidget.rowCount()):
            path_consumption_vector.append(
                float(self.HydraTableWidget.cellWidget(row, 4).value()))

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
        global incidence_matrix_tr
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
        global R_matrix
        matrix_size = len(R_factor_vector)
        diagonal_R_matrix = np.zeros((matrix_size, matrix_size))
        np.fill_diagonal(diagonal_R_matrix, R_factor_vector)
        R_matrix = diagonal_R_matrix
        print(R_matrix)

    def CalculateM0Matrix(self):
        global M0_matrix
        if len(incidence_matrix_tr) == 0 or len(R_matrix) == 0:
            QMessageBox().warning(
                None, "Матрица не существует",
                "Матрицы инцидентности или матрицы вектора R не сущесвует.")
            return
        incidence_matrix = incidence_matrix_tr.transpose()
        M0_matrix = np.matmul(
            np.matmul(incidence_matrix, R_matrix), incidence_matrix_tr)
        print("M0 matrix")
        print(M0_matrix)

    def CreatePressureArray(self):
        global pressure_vector
        pressure_vector = [3000 if obj_type ==
                           "ГРП" else 0 for obj_type in objects_dict.values()]

    def CreateNodesCount(self):
        global edge_nodes_count
        global inner_nodes_count
        edge_nodes_count = sum(
            1 for obj_type in objects_dict.values() if obj_type == "ГРП")
        inner_nodes_count = len(objects_name_list) - edge_nodes_count

    def ShiftArray(self, matrix, row_offset, col_offset, height, width):
        arr = np.array(matrix)
        if arr.ndim == 1:  # Проверяем, является ли массив одномерным
            start_index = max(0, col_offset)
            end_index = min(arr.shape[0], col_offset + width)
            shifted_arr = arr[start_index:end_index]
        else:
            rows, cols = arr.shape
            start_row = max(0, row_offset)
            end_row = min(rows, row_offset + height)
            start_col = max(0, col_offset)
            end_col = min(cols, col_offset + width)
            shifted_arr = arr[start_row:end_row, start_col:end_col]
        return shifted_arr

    def CalculateM0MultPVector(self):
        global m0_mult_press_vector
        m0_vector = self.ShiftArray(
            M0_matrix, 0, inner_nodes_count,
            inner_nodes_count, edge_nodes_count)
        P_vector = [value for value in pressure_vector if value != 0]
        m0_mult_press_vector = -np.dot(m0_vector, P_vector)
        print("M0 * Вектор давления")
        print(m0_mult_press_vector)

    def ObjectsNodeConsumption(self):
        col = 3
        beginning_object_array = []
        ending_object_array = []
        node_consumption_vector.clear()
        for row in range(self.HydraTableWidget.rowCount()):
            beginning_edge_text = str(
                self.HydraTableWidget.cellWidget(row, 1).currentText())
            end_edge_text = str(
                self.HydraTableWidget.cellWidget(row, 2).currentText())
            beginning_object_array.append(beginning_edge_text)
            ending_object_array.append(end_edge_text)

        for row, object_name in enumerate(objects_name_list):
            sum_beginning_node = sum(Q * 0.45 for beg_obj, Q in zip(
                beginning_object_array,
                path_consumption_vector) if beg_obj == object_name)
            sum_ending_node = sum(Q * 0.55 for end_obj, Q in zip(
                ending_object_array,
                path_consumption_vector) if end_obj == object_name)
            result = sum_beginning_node + sum_ending_node
            node_consumption_vector.append(result)
            item = QTableWidgetItem(str(result))
            self.ObjectsTableWidget.setItem(row, col, item)
        print("узловой расход")
        print(node_consumption_vector)

    def CalculateM0MultPVectorPlusQ(self):
        m0_mult_press_vector_plus_Q.clear()
        for index, vector in enumerate(m0_mult_press_vector):
            result = vector - node_consumption_vector[index]
            m0_mult_press_vector_plus_Q.append(result)
        print("m0p+Q")
        print(m0_mult_press_vector_plus_Q)

    def CalculateP0Vector(self):
        global P0_vector
        global p_k_vector
        matrix_1 = np.array(self.ShiftArray(
            M0_matrix, 0, 0, inner_nodes_count, inner_nodes_count))
        inverse_matrix_1 = np.linalg.inv(matrix_1)
        matrix_2 = m0_mult_press_vector_plus_Q
        P0_vector = np.dot(inverse_matrix_1, matrix_2)
        p_k_vector = P0_vector
        print("P0 vector")
        print(P0_vector)
        print("p(k) vector")
        print(p_k_vector)

    def CalculateYKVector(self):
        global y_k_vector
        matrix_1 = self.ShiftArray(
            incidence_matrix_tr, 0, 0, len(edges_vector), inner_nodes_count)
        matrix_2 = p_k_vector
        matrix_3 = self.ShiftArray(
            incidence_matrix_tr, 0, inner_nodes_count,
            len(edges_vector), edge_nodes_count)
        matrix_4 = [value for value in pressure_vector if value != 0]
        y_k_vector = np.dot(matrix_1, matrix_2) + np.dot(matrix_3, matrix_4)
        print("Y(K) vector")
        print(y_k_vector)

    def CalculateXKVector(self):
        x_k_vector.clear()
        for i, value in enumerate(y_k_vector):
            x_k_vector.append(math.sqrt(
                abs(value)/hydraulic_friction_factor[i]) * np.sign(value))
        print('X(k) vector')
        print(x_k_vector)

    def Calculate1SXKVector(self):
        s1_x_k_vector.clear()
        for i, value in enumerate(x_k_vector):
            s1_x_k_vector.append(
                1 / (hydraulic_friction_factor[i] * abs(value)))
        print("1/S(X)k vector")
        print(s1_x_k_vector)

    def Create1SXKMatrix(self):
        global s1_x_k_matrix
        matrix_size = len(s1_x_k_vector)
        diagonal_matrix = np.zeros((matrix_size, matrix_size))
        np.fill_diagonal(diagonal_matrix, s1_x_k_vector)
        s1_x_k_matrix = diagonal_matrix
        print("1/S(X)k matrix")
        print(s1_x_k_matrix)

    def CalculateAxKVector(self):
        global Ax_k_vector
        matrix_1 = incidence_matrix_tr.transpose()
        matrix_2 = x_k_vector
        Ax_k_vector = np.dot(matrix_1, matrix_2)
        print("A(x)K vector")
        print(Ax_k_vector)

    def CalculateSigmaGKVector(self):
        sigma_g_k_vector.clear()
        for i, value in enumerate(Ax_k_vector):
            sigma_g_k_vector.append(value + node_consumption_vector[i])
        print("Sigma G(K) vector")
        print(sigma_g_k_vector)

    def CalculateMKMatrix(self):
        global m_k_matrix
        matrix_1 = incidence_matrix_tr.transpose()
        matrix_2 = s1_x_k_matrix
        m_k_matrix = 0.5 * np.dot(
            np.dot(matrix_1, matrix_2), incidence_matrix_tr)
        print("M(k) matrix")
        print(m_k_matrix)

    def CalculateDeltaPKVector(self):
        global delta_p_k_vector
        matrix_1 = self.ShiftArray(
            m_k_matrix, 0, 0, inner_nodes_count, inner_nodes_count)
        inverse_matrix_1 = np.linalg.inv(matrix_1)
        matrix_2 = sigma_g_k_vector[:inner_nodes_count]
        delta_p_k_vector = -np.dot(inverse_matrix_1, matrix_2)
        print("Delta P(K) vector")
        print(delta_p_k_vector)

    def CalculateDeltaPKPlus1Vector(self):
        delta_p_k_plus_1_vector.clear()
        for i, value in enumerate(p_k_vector):
            delta_p_k_plus_1_vector.append(value + delta_p_k_vector[i])
        for i in range(edge_nodes_count):
            delta_p_k_plus_1_vector.append(3000)
        print("Delta P(k+1) vector")
        print(delta_p_k_plus_1_vector)

    def CalculateAll(self):
        self.CreateEdgesArray()
        self.CreatePathConsumptionArray()
        self.ObjectsNodeConsumption()

        self.CreatePressureArray()  # нужен полный список объектов
        self.CreateNodesCount()  # нужен полный список объектов
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
        self.CalculateM0Matrix()  # нужна матрица инцидентности и матрица R
        self.CalculateM0MultPVector()  # вектор P, матрица M0, гран и внут узлы
        self.CalculateM0MultPVectorPlusQ()  # M0P, Q
        self.CalculateP0Vector()  # M0P+Q, M0
        self.CalculateYKVector()
        self.CalculateXKVector()
        self.Calculate1SXKVector()
        self.Create1SXKMatrix()
        self.CalculateAxKVector()
        self.CalculateSigmaGKVector()
        self.CalculateMKMatrix()
        self.CalculateDeltaPKVector()
        self.CalculateDeltaPKPlus1Vector()

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
        filename = f"graphs_pic/topology_{len(graph.edges)}.png"
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

    def ClearHydraTable(self):
        self.HydraTableWidget.setRowCount(0)
        self.AddHydraRow()

    def HydraSaveToCSV(self):
        try:
            file_dialog = QFileDialog()
            path, _ = file_dialog.getSaveFileName(
                None, "Сохранить Гидравлику как файл CSV", "",
                "CSV Files (*.csv);;All Files (*)")
            if path:
                with open(path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(
                        ['№ участка', 'Начало участка', 'Конец участка',
                         'L, м', 'Q пут, н.м3/ч', 'Тип трубы',
                         'Диаметр трубы', 'Скорость газа V, м/с'])
                    for row in range(self.HydraTableWidget.rowCount()):
                        hydra_number = self.HydraTableWidget.cellWidget(
                            row, 0).value()
                        beginning_object = self.HydraTableWidget.cellWidget(
                            row, 1).currentText()
                        end_object = self.HydraTableWidget.cellWidget(
                            row, 2).currentText()
                        length = self.HydraTableWidget.cellWidget(
                            row, 3).value()
                        Q = self.HydraTableWidget.cellWidget(row, 4).value()
                        pipe_type = self.HydraTableWidget.cellWidget(
                            row, 5).currentText()
                        diameter = self.HydraTableWidget.item(row, 6).text()
                        gas_speed = self.HydraTableWidget.item(row, 7).text()
                        writer.writerow(
                            [hydra_number, beginning_object, end_object,
                             length, Q, pipe_type, diameter, gas_speed])
                QMessageBox().information(None, "Сохранено",
                                          f"Данные успешно сохранены в файл CSV: {path}")  # noqa E501
        except Exception as e:
            QMessageBox().critical(None, "Ошибка",
                                   f"Произошла ошибка при сохранении: {e}")

    def HydraLoadFromCSV(self):
        try:
            file_dialog = QFileDialog()
            path, _ = file_dialog.getOpenFileName(
                None, "Загрузить Гидравлику как файл CSV",
                "", "CSV Files (*.csv)")
            if path:
                with open(path, 'r', encoding='utf-8') as csvfile:
                    reader = csv.reader(csvfile)
                    self.ClearHydraTable()
                    for row, row_data in enumerate(reader):
                        if row > 0:
                            self.AddHydraRow()
                            for col, data in enumerate(row_data):
                                if col == 0:
                                    self.HydraTableWidget.cellWidget(
                                        row - 1, col).setValue(int(data))
                                elif col == 1:
                                    self.HydraTableWidget.cellWidget(
                                        row - 1, col).setCurrentText(str(data))
                                elif col == 2:
                                    self.HydraTableWidget.cellWidget(
                                        row - 1, col).setCurrentText(str(data))
                                elif col == 3:
                                    self.HydraTableWidget.cellWidget(
                                        row - 1, col).setValue(int(data))
                                elif col == 4:
                                    self.HydraTableWidget.cellWidget(
                                        row - 1, col).setValue(float(data))
                                elif col == 5:
                                    self.HydraTableWidget.cellWidget(
                                        row - 1, col).setCurrentText(str(data))
                                elif col == 6:
                                    item = QTableWidgetItem(data)
                                    self.HydraTableWidget.setItem(
                                        row - 1, col, item)
                                elif col == 7:
                                    item = QTableWidgetItem(data)
                                    self.HydraTableWidget.setItem(
                                        row - 1, col, item)
            self.RemoveHydraRow()
            QMessageBox().information(None, "Импорт завершен",
                                      "Данные успешно импортированы.")
        except Exception as e:
            QMessageBox().critical(None, "Ошибка",
                                   f"Произошла ошибка при загрузке: {e}")
