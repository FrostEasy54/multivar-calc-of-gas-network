import numpy as np
import networkx as nx
import math
from prettytable import PrettyTable

# Создаем граф
G = nx.MultiDiGraph()

# Добавляем узлы
nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 'ГРП']
G.add_nodes_from(nodes)

# Добавляем ребра
edges = [(2, 1), (13, 16,), (6, 5), (2, 3), (4, 3), (8, 4),
         (8, 7), (6, 7), (10, 6), (10, 9), (13, 8),
         (12, 13), ('ГРП', 12), (11, 10), (10, 14),
         (15, 17), (11, 15), (16, 18),
         (15, 16), (15, 14), (6, 2), (12, 11)]
G.add_edges_from(edges)


# 2.1 Транспонированная матрица инцидентности !
def create_incidence_matrix(nodes, edges):
    # Создаем матрицу инцидентности
    incidence_matrix = np.zeros((len(nodes), len(edges)), dtype=int)
    edge_indices = {edge: i for i, edge in enumerate(edges)}
    node_indices = {node: i for i, node in enumerate(nodes)}

    for edge in edges:
        node1, node2 = edge
        edge_index = edge_indices[edge]
        node1_index = node_indices[node1]
        node2_index = node_indices[node2]
        incidence_matrix[node1_index, edge_index] = 1
        incidence_matrix[node2_index, edge_index] = -1

    incidence_matrix_tr = incidence_matrix.transpose()
    # Подготовим данные для PrettyTable
    table = PrettyTable()
    table.field_names = ['Участки/Узлы'] + nodes
    for i, row in enumerate(incidence_matrix_tr):
        table.add_row([f"{edges[i]}"] + list(row))

    # Выводим таблицу с использованием PrettyTable
    # print(table)
    return incidence_matrix_tr


incidence_matrix_tr = create_incidence_matrix(nodes, edges)  # работает
# print(incidence_matrix_tr)

inner_diameteres = {(2, 1): 51, (13, 16,): 151, (6, 5): 51, (2, 3): 54,
                    (4, 3): 100, (8, 4): 100, (8, 7): 151, (6, 7): 151,
                    (10, 6): 207, (10, 9): 51, (13, 8): 207,
                    (12, 13): 207, ('ГРП', 12): 618, (11, 10): 207,
                    (10, 14): 70, (15, 17): 70, (11, 15): 151, (16, 18): 70,
                    (15, 16): 151, (15, 14): 70, (6, 2): 100, (12, 11): 207}

Q_edge = [18.6, 151.1, 41.3, 25.8, 77.5, 53, 177.1, 91.6, 174, 22.8,
          107.5, 172, 0, 119.8, 86.2, 43, 227, 43, 189.7, 50.5, 88.8, 34.4]


# 2.2
def calculate_velocity(diameter, Q):
    area = (0.25 * math.pi * (diameter / 1000)
            ** 2)  # переводим диаметр в метры
    # переводим расход в м3/ч и вычисляем скорость в м/с
    velocity = Q / (area * 3600)
    return velocity


edge_velocitices_vector = []
for i in range(len(Q_edge)):
    diameter_key = tuple(inner_diameteres.keys())[i]
    diameter = inner_diameteres[diameter_key]
    Q = Q_edge[i]
    velocity = calculate_velocity(diameter, Q)
    edge_velocitices_vector.append(velocity)

# print(edge_velocitices_vector) # работает


# 2.3
Reynolds_number_vector = []
natural_gas_viscosity = 0.0000143
diameter_values = list(inner_diameteres.values())  # список удобный!
for i in range(len(Q_edge)):
    Reynolds_number = edge_velocitices_vector[i] * \
        diameter_values[i] / 1000 / natural_gas_viscosity
    Reynolds_number_vector.append(Reynolds_number)
# print(Reynolds_number_vector) # работает

# 2.4 есть ещё несколько типов труб без коэффициентов, пока не добавлял
roughness_factor_dict = {'Сталь': 0.1000,
                         'OK': 0.1500, 'REHAU': 0.0070, 'МП': 0.0004}

# 2.5
friction_factor_vector = []


def calculate_friction_factor(Reynolds_number, inner_diameter, roughness_factor):
    Reynolds_roughness_diameter_ratio = Reynolds_number * \
        roughness_factor / inner_diameter
    if Reynolds_number == 0:
        friction_factor = 1 / \
            ((2 * math.log10(3.7/(roughness_factor/inner_diameter)))**2)
        friction_factor_vector.append(friction_factor)
    elif Reynolds_number < 2300:
        friction_factor = 64 / Reynolds_number
        friction_factor_vector.append(friction_factor)
    elif Reynolds_roughness_diameter_ratio < 23 and Reynolds_number < 125000:
        friction_factor = 0.3164 / (Reynolds_number ** 0.25)
        friction_factor_vector.append(friction_factor)
    elif Reynolds_roughness_diameter_ratio < 23 and Reynolds_number > 125000:
        friction_factor = 0.0032 + (0.221 / (Reynolds_number ** 0.237))
        friction_factor_vector.append(friction_factor)
    elif Reynolds_roughness_diameter_ratio >= 23 and Reynolds_roughness_diameter_ratio < 560:
        friction_factor = 0.11 * \
            (((roughness_factor/inner_diameter) + (68/Reynolds_number)) ** 0.25)
        friction_factor_vector.append(friction_factor)
    elif Reynolds_roughness_diameter_ratio >= 560:
        friction_factor = 1 / \
            ((2 * math.log10(3.7/(roughness_factor/inner_diameter)))**2)
        friction_factor_vector.append(friction_factor)


for i in range(len(Q_edge)):
    calculate_friction_factor(
        Reynolds_number_vector[i], diameter_values[i], roughness_factor_dict.get('Сталь'))
# print(friction_factor_vector) # работает!


gas_density = 0.73
pipe_length_vector = [100, 450, 100, 100, 300, 205, 290, 150, 300,
                      100, 305, 250, 10, 200, 350, 150, 390, 150, 305, 205, 200, 50]
hydraulic_friction_factor_vector = []


def calculate_hydraulic_friction_factor(friction_factor, pipe_length, pipe_diameter):
    hydraulic_friction_factor = (gas_density / 2) * friction_factor * pipe_length / (
        pipe_diameter/1000) * 16/((3600*math.pi*((pipe_diameter/1000)**2))**2)
    hydraulic_friction_factor_vector.append(hydraulic_friction_factor)


for i in range(len(Q_edge)):
    calculate_hydraulic_friction_factor(
        friction_factor_vector[i], pipe_length_vector[i], diameter_values[i])
# print(hydraulic_friction_factor_vector) # работает !


R_factor_vector = []


def calculate_R_factor(hydraulic_friction_factor):
    R_factor = 1 / ((hydraulic_friction_factor) ** 0.5)
    R_factor_vector.append(R_factor)


for i in range(len(Q_edge)):
    calculate_R_factor(hydraulic_friction_factor_vector[i])
# print(R_factor_vector) # работает !


def create_diagonal_matrix(matrix):
    matrix_size = len(matrix)
    diagonal_matrix = np.zeros((matrix_size, matrix_size))
    np.fill_diagonal(diagonal_matrix, matrix)
    return diagonal_matrix


# 3.1
R_factor_matrix = create_diagonal_matrix(R_factor_vector)
# print(R_factor_matrix)  # работает !


def print_matrix(matrix):
    table = PrettyTable()
    table.field_names = [index for index in range(1, len(matrix) + 1)]
    for row in matrix:
        formatted_row = ["{:.2f}".format(element) for element in row]
        table.add_row(formatted_row)
    # print(table)


# print_matrix(R_factor_matrix)

# 3.2
def calculate_M0_matrix(transposed_incidence_matrix, R_factor_matrix):
    incidence_matrix = transposed_incidence_matrix.transpose()
    M0_matrix = np.matmul(
        np.matmul(incidence_matrix, R_factor_matrix), transposed_incidence_matrix)
    return M0_matrix


M0_matrix = calculate_M0_matrix(
    incidence_matrix_tr, R_factor_matrix)  # работает!
# print(M0_matrix)
# print_matrix(M0_matrix)


# вектор давлений
pressure_vector = [3000 if node == "ГРП" else 0 for node in nodes]
# print('Давление')
# print(pressure_vector)

# количество граничных узлов
edge_nodes_count = sum(1 for node in nodes if node == "ГРП")
inner_nodes_count = len(nodes) - edge_nodes_count
# Cмещение матрицы


def ShiftArray(matrix, row_offset, col_offset, height, width):
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
# print(shift_array(M0_matrix, 0, 17, 17, 2))


def m0_vector_mult_P_vector(m0_matrix, pressure_vector, edge_nodes_count, inner_nodes_count):
    m0_vector = ShiftArray(m0_matrix, 0, inner_nodes_count,
                           inner_nodes_count, edge_nodes_count)
    P_vector = [value for value in pressure_vector if value != 0]
    result = -np.dot(m0_vector, P_vector)
    return result

# print(m0_vector_mult_P_vector(M0_matrix, pressure_vector, edge_nodes_count, inner_nodes_count)) # работает !!!

# 3.3


def calculate_1_S_Xk_matrix():
    pass


p_k_vector = [2424.4607741365794, 2456.72499555618, 2390.2209687400755,
              2432.70234199842, 2463.314096129302, 2622.3867051973994,
              2614.00556846254, 2630.596341823753,
              2663.699348740133, 2712.179614848253, 2873.274155383222,
              2999.7348758605795, 2701.100477324184, 2342.5799183940235,
              2469.7535444744763, 2469.57415046951, 2420.847493459498, 3000]


def CalculateYKVector():
    global y_k_vector
    matrix_1 = ShiftArray(
        incidence_matrix_tr, 0, 0, len(Q_edge), inner_nodes_count)
    matrix_2 = p_k_vector[:inner_nodes_count]
    print("матрица 1")
    print(matrix_1)
    print("матрица 2")
    print(matrix_2)
    matrix_3 = ShiftArray(
        incidence_matrix_tr, 0, inner_nodes_count,
        len(Q_edge), edge_nodes_count)
    matrix_4 = [value for value in pressure_vector if value != 0]
    print("матрица 3")
    print(matrix_3)
    print("матрица 4")
    print(matrix_4)
    result_1 = np.dot(matrix_1, matrix_2)
    result_2 = np.dot(matrix_3, matrix_4)
    print("Результ 1")
    print(result_1)
    print("Результ 2")
    print(result_2)
    y_k_vector = result_1 + result_2
    print("Y(K) vector")
    print(y_k_vector)


CalculateYKVector()