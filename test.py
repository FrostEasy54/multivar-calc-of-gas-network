import networkx as nx
import pygraphviz as pgv

# Создаем ориентированный граф
G = nx.MultiGraph()

# Добавляем узлы
G.add_nodes_from(['П1', 2, 3, 4, 'П2', 6, 7, 8, 'П3', 10,
                 11, 'ГРП', 13, 14, 15, 16, 'П4', 'П5'])

# Добавляем ребра
G.add_edges_from([('П1', 2), (2, 3), (3, 4), ('П2', 6), (6, 7), (7, 8), ('П3', 10),
                 (10, 11), (11, 'ГРП'), ('ГРП', 13),
                  (10, 14), (11, 15), (13, 16),
                  (14, 15), (15, 16), (2, 6), (6, 10),
                  (15, 'П4'), (16, 'П5'), (4, 8), (8, 13)])

# Визуализация графа
A = nx.nx_agraph.to_agraph(G)
A.layout('dot')  # twopi неплохой, dot тоже
A.draw('multigraph.png')


# проверка связности графа
def is_graph_connected():
    is_connected = nx.is_connected(G)
    if is_connected:
        print("Граф связный")
    else:
        print("Граф несвязный")


# Минимальное остовное дерево
def min_span_tree():
    mst = nx.minimum_spanning_tree(G)
    B = nx.nx_agraph.to_agraph(mst)
    B.layout('dot')
    B.draw('min_span_tree.png')


# Вроде более менее, но здесь тупиковая система получается
def MVR():
    start_node = 'ГРП'
    end_nodes = [node for node in G.nodes() if isinstance(node, str)
                 and node.startswith('П')]

    all_paths = {}
    for end_node in end_nodes:
        paths = list(nx.all_simple_paths(G, start_node, end_node))
        all_paths[end_node] = paths

    # Выбираем самый короткий путь для каждой вершины "П"
    selected_paths = {end_node: min(paths, key=len)
                      for end_node, paths in all_paths.items()}

    # Создаем новый граф на основе выбранных путей
    new_G = nx.Graph()
    for path in selected_paths.values():
        new_G.add_edges_from(zip(path[:-1], path[1:]))
    # Визуализация графа с помощью PyGraphviz
    A = nx.nx_agraph.to_agraph(new_G)
    A.layout('dot')
    A.draw('graph_with_shortest_paths.png')


#MVR()
is_graph_connected()
