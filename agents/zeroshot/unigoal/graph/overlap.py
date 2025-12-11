import json
import networkx as nx
from networkx.algorithms import isomorphism
import numpy as np
from scipy.spatial.distance import cosine
from heapq import heappush, heappop
from grakel import Graph, kernels


class GraphMatcher:
    def __init__(self, graph1, graph2, llm=None):
        self.graph1 = graph1
        self.graph2 = graph2
        self.llm = llm

        G1 = nx.DiGraph()
        for node in graph1['nodes']:
            G1.add_node(node['id'], position=node['position'])
        for edge in graph1['edges']:
            G1.add_edge(edge['source'], edge['target'], type=edge['type'])

        G2 = nx.DiGraph()
        for node in graph2['nodes']:
            G2.add_node(node['id'])  # No position assigned initially
        for edge in graph2['edges']:
            G2.add_edge(edge['source'], edge['target'], type=edge['type'])
        self.G1 = G1
        self.G2 = G2

    def node_similarity(self, G1, G2):
        nodes_G1 = set(G1.nodes)
        nodes_G2 = set(G2.nodes)
        common_nodes = nodes_G1.intersection(nodes_G2)

        if len(nodes_G2) == 0:
            node_overlap_ratio = 0.0
        else:
            node_overlap_ratio = len(common_nodes) / len(nodes_G2)

        degree_sim = 0.0
        if common_nodes:
            for node in common_nodes:
                degree_G1 = G1.degree(node)
                degree_G2 = G2.degree(node)
                degree_diff = abs(degree_G1 - degree_G2)
                max_degree = max(degree_G1, degree_G2)
                if max_degree > 0:
                    degree_sim += 1.0 - (degree_diff / max_degree)
            degree_sim /= len(common_nodes)

        node_sim = (node_overlap_ratio + degree_sim) / 2.0
        return node_sim

    def edge_similarity(self, G1, G2):
        edges_G1 = {frozenset((u, v, frozenset(data.items()))) for u, v, data in G1.edges(data=True)}
        edges_G2 = {frozenset((u, v, frozenset(data.items()))) for u, v, data in G2.edges(data=True)}
        common_edges = edges_G1.intersection(edges_G2)

        if len(edges_G2) == 0:
            return 1.0
        else:
            return len(common_edges) / len(edges_G2)

    def graph_edit_distance_heuristic(self, G1, G2):
        node_diff = abs(len(G1) - len(G2))
        edge_diff = abs(G1.number_of_edges() - G2.number_of_edges())
        return node_diff + edge_diff

    def apply_operation(self, operation, G, **kwargs):
        if operation == 'add_node':
            G.add_node(kwargs['node'])
        elif operation == 'remove_node':
            G.remove_node(kwargs['node'])
        elif operation == 'add_edge':
            G.add_edge(*kwargs['edge'])
        elif operation == 'remove_edge':
            G.remove_edge(*kwargs['edge'])

    def overlap(self):
        G1 = self.G1
        G2 = self.G2

        self.common_nodes = self.find_common_nodes(G1, G2)
        node_sim = self.node_similarity(G1, G2)
        edge_sim = self.edge_similarity(G1, G2)

        combined_sim = (node_sim + edge_sim) / 2
        return combined_sim

    def find_common_nodes(self, G1, G2):
        """ Find common nodes between two graphs """
        return set(G1.nodes).intersection(set(G2.nodes))

    def calculate_relative_positions(self, graph, common_nodes):
        """ Calculate relative positions of nodes within a graph using LLM for unknown positions """
        relative_positions = {}
        if not common_nodes:
            return relative_positions

        ref_node = list(common_nodes)[0]

        for node in common_nodes:
            if node == ref_node:
                continue
            prompt = f"Given the following information: {ref_node} and {node}. Please provide the relative position of {node} with respect to {ref_node} in the format [x, y]."
            response = self.llm(prompt)
            try:
                rel_pos_str = response.split("[")[1].split("]")[0]
                rel_pos = [float(coord.strip()) for coord in rel_pos_str.split(",")]
                key = tuple((node, ref_node))
                relative_positions[key] = rel_pos
            except (IndexError, ValueError) as e:
                pass

        positions = {}
        ref_node = next(iter(relative_positions))[1]
        positions[ref_node] = [0, 0]
        for node_c in relative_positions.keys():
            positions[node_c[0]] = relative_positions[node_c]

        return positions

    def predict_remaining_node_positions(self, common_nodes, positions, scene_graph):
        if len(common_nodes) < 2:
            raise ValueError("At least two common nodes are required to predict the position of other nodes")

        ref_points = sorted(list(common_nodes))[:2]
        ref_point_1, ref_point_2 = ref_points

        ref_pos_1 = np.array(scene_graph.nodes[ref_point_1]['position'])
        ref_pos_2 = np.array(scene_graph.nodes[ref_point_2]['position'])

        ref_vec_scene = ref_pos_2 - ref_pos_1
        try:
            ref_vec_subgraph = np.array(positions[ref_point_1]) - np.array(positions[ref_point_2])
        except KeyError as e:
            print(f"KeyError: {e}")
            return {}

        if np.allclose(ref_vec_subgraph, 0):
            ref_vec_subgraph = np.array([1000., 1000.])

        angle = np.arctan2(ref_vec_scene[1], ref_vec_scene[0]) - np.arctan2(ref_vec_subgraph[1], ref_vec_subgraph[0])
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])

        scale_factor = np.linalg.norm(ref_vec_scene) / np.linalg.norm(ref_vec_subgraph)

        predicted_positions = {}
        for node, rel_pos in positions.items():
            if node not in ref_points:
                relative_position = np.array(rel_pos) - np.array(positions[ref_point_1])
                transformed_pos = np.dot(rotation_matrix, np.array(relative_position) * scale_factor) + ref_pos_1
                predicted_positions[node] = transformed_pos

        if len(predicted_positions) > 0:
            rel_pos_list = []
            for node, rel_pos in predicted_positions.items():
                rel_pos_list.append(rel_pos)
            position = sum(rel_pos_list) / len(rel_pos_list)
        else:
            rel_pos_list = [ref_pos_1, ref_pos_2]
            position = sum(rel_pos_list) / len(rel_pos_list)
        position = list(position)
        return position
