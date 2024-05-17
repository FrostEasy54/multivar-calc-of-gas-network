from PyQt6.QtWidgets import QTableWidgetItem, QComboBox
from PyQt6.QtWidgets import QDoubleSpinBox, QFileDialog
from PyQt6.QtWidgets import QSpinBox, QMessageBox, QDialog, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QColor
from PyQt6 import QtWidgets

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
p_k_plus_1_vector = []
p_k_vector = []
q_k_vector = []
q_k_plus_1_vector = []
proportion_q_k_pl_1_to_q_k = []
variant_data = {}
result_variant_data = {}


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
        # Привязываем обработчики событий для новой строки
        self.bindEventHandlersForRow(self.HydraTableWidget.rowCount() - 1)

    def bindEventHandlersForRow(self, row):
        self.HydraNumberSpinBox(row)
        self.HydraBeginningComboBox(row)
        self.HydraEndComboBox(row)
        self.HydraLengthSpinBox(row)
        self.HydraPathConsumptionDoubleSpinBox(row)
        self.HydraPipeTypeComboBox(row)
        self.HydraPipeDiameter(row)

    def RemoveHydraRow(self):
        if self.HydraTableWidget.rowCount() == 1:
            QMessageBox().warning(
                None, "Единственная строка", "Вы не можете удалить единственную строку в таблице.")  # noqa E501
            return
        else:
            self.HydraTableWidget.removeRow(
                self.HydraTableWidget.rowCount()-1)

    def HydraNumberSpinBox(self, row):
        # Указываем столбец, для которого нужно установить SpinBox
        col = 0
        current_row = row
        # Получаем значение номера помещения из предыдущей строки
        prev_row = self.HydraTableWidget.rowCount() - 2
        prev_widget = self.HydraTableWidget.cellWidget(prev_row, col)
        prev_value = prev_widget.value() if prev_widget else 0
        sb = QSpinBox()
        sb.setMaximum(1000)
        sb.setValue(prev_value + 1)
        self.HydraTableWidget.setCellWidget(current_row, col, sb)

    def HydraBeginningComboBox(self, row):
        col = 1
        cb = QComboBox()
        cb.addItems(objects_name_list)
        self.HydraTableWidget.setCellWidget(row, col, cb)

    def HydraEndComboBox(self, row):
        col = 2
        cb = QComboBox()
        cb.addItems(objects_name_list)
        self.HydraTableWidget.setCellWidget(row, col, cb)

    def HydraLengthSpinBox(self, row):
        col = 3
        sb = QSpinBox()
        sb.setMaximum(1000)
        sb.setMinimum(0)
        self.HydraTableWidget.setCellWidget(row, col, sb)

    def HydraPathConsumptionDoubleSpinBox(self, row):
        col = 4
        sb = QDoubleSpinBox()
        sb.setMaximum(1000)
        sb.setMinimum(0)
        self.HydraTableWidget.setCellWidget(row, col, sb)

    def HydraPipeTypeComboBox(self, row):
        col = 5
        cb = QComboBox()
        cb.addItems(pipe_type_dict.keys())
        self.HydraTableWidget.setCellWidget(row, col, cb)
        # Передаем номер строки в лямбда-функцию
        cb.currentTextChanged.connect(
            lambda text, row=row: self.HydraPipeDiameter(row))

    def HydraPipeDiameter(self, row):
        col = 6
        combo_box_item = self.HydraTableWidget.cellWidget(row, 5)
        selected_pipe = combo_box_item.currentText()
        diameter = pipe_type_dict.get(selected_pipe, "Нет данных")
        diameter_item = QTableWidgetItem(str(diameter))
        self.HydraTableWidget.setItem(row, col, diameter_item)

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

    def CreateVelocityArray(self, Q_array):
        gas_velocity_vector.clear()
        for i in range(len(edges_vector)):
            area = (0.25 * math.pi * (pipe_diameter_vector[i] / 1000) ** 2)
            velocity = Q_array[i] / (area * 3600)
            gas_velocity_vector.append(velocity)
        print("V вектор")
        print(gas_velocity_vector)

    def CreateDiameterArray(self):
        pipe_diameter_vector.clear()
        for row in range(self.HydraTableWidget.rowCount()):
            diameter = float(self.HydraTableWidget.item(row, 6).text())
            pipe_diameter_vector.append(diameter)

    def CreateLengthArray(self):
        pipe_length_vector.clear()
        for row in range(self.HydraTableWidget.rowCount()):
            length = int(self.HydraTableWidget.cellWidget(row, 3).value())
            pipe_length_vector.append(length)

    def CalculateReynoldsNumber(self):
        Reynolds_number_vector.clear()
        for i in range(len(edges_vector)):
            Reynolds_number = gas_velocity_vector[i] * \
                pipe_diameter_vector[i] / 1000 / NATURAL_GAS_VISCOSITY
            Reynolds_number_vector.append(Reynolds_number)

    def CreatePipeRoughnessFactor(self):
        roughness_factor_vector.clear()
        for i in range(len(edges_vector)):
            material_name = self.HydraTableWidget.cellWidget(
                i, 5).currentText().split()[0]
            if material_name in roughness_factor_dict:
                roughness_factor_vector.append(
                    roughness_factor_dict[material_name])
            else:
                roughness_factor_vector.append(0)

    def CalculateDarsiFrictionFactor(self):
        Darsi_friction_factor_vector.clear()
        for i in range(len(edges_vector)):
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
            if hydraulic_friction_factor[i] < 0:
                QMessageBox().warning(None, "Ошибка вычисления", "Невозможно вычислить корень из отрицательного числа.")  # noqa E501
                return
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

        '''        table = PrettyTable()
        table.field_names = ['Участки/Узлы'] + objects_name_list
        for i, row in enumerate(incidence_matrix_tr):
            table.add_row([f"{edges_vector[i]}"] + list(row))
        print("Матрица инцидентности")
        # Выводим таблицу с использованием PrettyTable
        print(table)'''

    def CreateRFactorMatrix(self):
        global R_matrix
        matrix_size = len(R_factor_vector)
        diagonal_R_matrix = np.zeros((matrix_size, matrix_size))
        np.fill_diagonal(diagonal_R_matrix, R_factor_vector)
        R_matrix = diagonal_R_matrix

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
            item = QTableWidgetItem(f"{result:.1f}")
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
        matrix_1 = np.array(self.ShiftArray(
            M0_matrix, 0, 0, inner_nodes_count, inner_nodes_count))
        inverse_matrix_1 = np.linalg.inv(matrix_1)
        matrix_2 = m0_mult_press_vector_plus_Q
        P0_vector = np.dot(inverse_matrix_1, matrix_2)
        print("P0 vector")
        print(P0_vector)

    def CalculateYKVector(self):
        global y_k_vector
        matrix_1 = self.ShiftArray(
            incidence_matrix_tr, 0, 0, len(edges_vector), inner_nodes_count)
        matrix_2 = p_k_vector[:inner_nodes_count]
        result_1 = np.dot(matrix_1, matrix_2)
        matrix_3 = self.ShiftArray(
            incidence_matrix_tr, 0, inner_nodes_count,
            len(edges_vector), edge_nodes_count)
        matrix_4 = [value for value in pressure_vector if value != 0]
        result_2 = np.dot(matrix_3, matrix_4)
        y_k_vector = result_1 + result_2
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

    def CalculatePKPlus1Vector(self):
        global delta_p_k_vector
        print("P(K) vector")
        print(p_k_vector)
        p_k_plus_1_vector.clear()
        for i, value in enumerate(delta_p_k_vector):
            p_k_plus_1_vector.append(value + p_k_vector[i])
        print("P(k+1) vector")
        print(p_k_plus_1_vector)

    def CalculateEpsG(self):
        global eps_g
        sigma_g_k_data = np.array(sigma_g_k_vector[:inner_nodes_count])
        min_sigma = np.min(sigma_g_k_data)
        max_sigma = np.max(sigma_g_k_data)
        eps_g = max(abs(min_sigma), max_sigma)
        print("eps G")
        print(eps_g)

    def CreateQKPlus1Array(self):
        q_k_plus_1_vector.clear()
        for value in x_k_vector:
            q_k_plus_1_vector.append(value)

    def CreateinitialBasic(self):
        self.CreateEdgesArray()
        self.CreateIncidenceMatrix()  # нужно чтоб существовал Edges Array
        self.CreatePathConsumptionArray()
        self.ObjectsNodeConsumption()
        self.CreatePressureArray()  # нужен полный список объектов
        self.CreateNodesCount()  # нужен полный список объектов
        self.CreateDiameterArray()
        self.CreateLengthArray()
        self.CreatePipeRoughnessFactor()

    def CalculateinitialApprox(self):
        global P0_vector
        Q = np.zeros(len(edges_vector))
        self.CreateVelocityArray(Q)
        self.CalculateReynoldsNumber()
        self.CalculateDarsiFrictionFactor()
        self.CalculateHydraulicFrictionFactor()
        self.CalculateRFactor()
        self.CreateRFactorMatrix()  # нужно чтоб существовал R Factor
        self.CalculateM0Matrix()  # нужна матрица инцидентности и матрица R
        self.CalculateM0MultPVector()   # вектор P, матрица M0, гран и внут узл
        self.CalculateM0MultPVectorPlusQ()  # M0P, Q
        self.CalculateP0Vector()  # M0P+Q, M0
        for value in P0_vector:
            p_k_vector.append(value)
        self.CalculateYKVector()  # нужна матрица инцидент и P(K) вектор
        self.CalculateXKVector()  # нужен вектор гидро давления и y(k)
        self.Calculate1SXKVector()  # нужен вектор гидро давления и x(k)
        self.Create1SXKMatrix()  # нужен 1/Sx(K) вектор
        self.CalculateAxKVector()  # инцидент и x(k)
        self.CalculateSigmaGKVector()  # A(x)KV и node_consumpt
        self.CalculateMKMatrix()  # инцидент и 1/Sx(K) матр
        self.CalculateDeltaPKVector()  # M(k) матрица и sigmaG(k) вектор
        self.CalculatePKPlus1Vector()  # p(k) вектор и delta
        self.CalculateEpsG()  # sigmaG(k)
        self.CreateQKPlus1Array()  # X(k)

    def CalculateIterations(self, Q):
        self.CreateVelocityArray(Q)
        self.CalculateReynoldsNumber()
        self.CalculateDarsiFrictionFactor()
        self.CalculateHydraulicFrictionFactor()
        self.CalculateRFactor()
        self.CreateRFactorMatrix()  # нужно чтоб существовал R Factor
        self.CalculateM0Matrix()  # нужна матрица инцидентности и матрица R
        self.CalculateM0MultPVector()   # вектор P, матрица M0, гран и внут узл
        self.CalculateM0MultPVectorPlusQ()  # M0P, Q
        self.CalculateP0Vector()  # M0P+Q, M0
        self.CalculateYKVector()  # нужна матрица инцидент и P(K) вектор
        self.CalculateXKVector()  # нужен вектор гидро давления и y(k)
        self.Calculate1SXKVector()  # нужен вектор гидро давления и x(k)
        self.Create1SXKMatrix()  # нужен 1/Sx(K) вектор
        self.CalculateAxKVector()  # инцидент и x(k)
        self.CalculateSigmaGKVector()  # A(x)KV и node_consumpt
        self.CalculateMKMatrix()  # инцидент и 1/Sx(K) матр
        self.CalculateDeltaPKVector()  # M(k) матрица и sigmaG(k) вектор
        self.CalculatePKPlus1Vector()  # p(k) вектор и delta
        self.CalculateEpsG()  # sigmaG(k)
        self.CreateQKPlus1Array()  # X(k)

    def IterationProcess(self):
        global q_0
        global p_k_vector
        global proportion_q_k_pl_1_to_q_k
        eps = 1
        count = 1
        has_greater_than_5 = True
        Q_0 = np.zeros(len(edges_vector))
        # Первичный итерационный процесс
        while eps > 0.1:
            print(f"Начало итерации--{count}")
            p_k_vector.clear()
            p_k_vector = p_k_plus_1_vector.copy()
            self.CalculateIterations(Q_0)
            eps = eps_g
            print(f"Конец итерации--{count}")
            count += 1

        count = 1

        # Вторичный итерационный процесс
        while has_greater_than_5:
            if count == 1000:
                break
            q_k_vector[:] = q_k_plus_1_vector
            proportion_q_k_pl_1_to_q_k.clear()
            proportion_q_k_pl_1_to_q_k = [
                abs(1 - q_k_plus_1_vector[i] / q_k_vector[i]) for i in range(
                    len(q_k_vector))]
            self.CalculateIterations(q_k_vector)
            eps = 1

            while eps > 0.1:
                if count == 1000:
                    break
                print(f"Начало итерации--{count}")
                p_k_vector.clear()
                p_k_vector = p_k_plus_1_vector.copy()
                self.CalculateIterations(q_k_vector)
                eps = eps_g
                print(f"Конец итерации--{count}")
                count += 1

            has_greater_than_5 = self.HasValueGreaterThan5(
                proportion_q_k_pl_1_to_q_k)

    def HasValueGreaterThan5(self, vector_1):
        return any(v1 > 0.5 for v1 in vector_1)

    def CalculateAll(self):
        self.CreateinitialBasic()
        self.CalculateinitialApprox()
        self.IterationProcess()
        self.HydraGasVelocity()
        self.ObjectsPressure()

    def HydraGasVelocity(self):
        col = 7
        max_index = gas_velocity_vector.index(max(gas_velocity_vector))
        for row in range(self.HydraTableWidget.rowCount()):
            item = QTableWidgetItem(f"{gas_velocity_vector[row]:.3f}")
            if row == max_index:
                item.setBackground(QColor(255, 0, 0))
            self.HydraTableWidget.setItem(row, col, item)

    def ObjectsPressure(self):
        col = 4
        min_index = p_k_plus_1_vector.index(min(p_k_plus_1_vector))
        for row in range(self.ObjectsTableWidget.rowCount()):
            if self.ObjectsTableWidget.cellWidget(
                    row, 1).currentText() == "ГРП":
                item = QTableWidgetItem("3000")
            else:
                if row < len(p_k_plus_1_vector):
                    item = QTableWidgetItem(f"{p_k_plus_1_vector[row]:.1f}")
                else:
                    item = QTableWidgetItem("")
            if row == min_index:
                item.setBackground(QColor(255, 0, 0))
            self.ObjectsTableWidget.setItem(row, col, item)

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
        edges_vector.clear()
        path_consumption_vector.clear()
        node_consumption_vector.clear()
        gas_velocity_vector.clear()
        pipe_diameter_vector.clear()
        pipe_length_vector.clear()
        Reynolds_number_vector.clear()
        roughness_factor_vector.clear()
        Darsi_friction_factor_vector.clear()
        hydraulic_friction_factor.clear()
        R_factor_vector.clear()
        m0_mult_press_vector_plus_Q.clear()
        x_k_vector.clear()
        s1_x_k_vector.clear()
        sigma_g_k_vector.clear()
        p_k_plus_1_vector.clear()
        p_k_vector.clear()
        proportion_q_k_pl_1_to_q_k.clear()
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
                        writer.writerow(
                            [hydra_number, beginning_object, end_object,
                             length, Q, pipe_type])
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
            self.RemoveHydraRow()
            QMessageBox().information(None, "Импорт завершен",
                                      "Данные успешно импортированы.")
        except Exception as e:
            QMessageBox().critical(None, "Ошибка",
                                   f"Произошла ошибка при загрузке: {e}")

    def HydraAddVariant(self):
        current_text = self.VariantComboBox.currentText()
        if current_text.startswith("Вариант"):
            variant_number = int(current_text.split(" ")[1])
            new_variant_text = f"Вариант {variant_number + 1}"
            if new_variant_text not in [
                    self.VariantComboBox.itemText(i) for i in range(
                        self.VariantComboBox.count())]:
                self.VariantComboBox.addItem(new_variant_text)

    def HydraDropVariant(self):
        current_index = self.VariantComboBox.currentIndex()
        if current_index != 0:
            self.VariantComboBox.removeItem(current_index)

    def SaveVariantData(self):
        current_variant = self.VariantComboBox.currentText()
        for row in range(self.HydraTableWidget.rowCount()):
            row_data = []
            for col in range(self.HydraTableWidget.columnCount()):
                item = self.HydraTableWidget.item(row, col)
                widget = self.HydraTableWidget.cellWidget(row, col)
                if widget:
                    if isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):  # noqa E501
                        row_data.append(widget.value())
                    elif isinstance(widget, QtWidgets.QComboBox):
                        row_data.append(widget.currentText())
                elif item is not None and item.text():
                    row_data.append(item.text())
            if row_data:
                variant_data.setdefault(current_variant, []).append(row_data)
        print(variant_data)
        self.CreateResultVariantData()

    def LoadVariantData(self):
        current_variant = self.VariantComboBox.currentText()
        if current_variant in variant_data:
            data = variant_data[current_variant]
            self.ClearHydraTable()
        # Заполняем таблицу данными
            for row, row_data in enumerate(data):
                self.AddHydraRow()

    def CreateResultVariantData(self):
        for variant in variant_data:
            print(variant)
