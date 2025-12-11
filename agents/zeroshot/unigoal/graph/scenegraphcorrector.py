from collections import deque

class SceneGraphCorrector:
    def __init__(self, llm_function):
        self.llm = llm_function

    def find_nodes_within_distance(self, graph, start_node, max_distance=2):
        visited = set()
        queue = deque([(start_node, 0)])
        nodes_within_distance = []

        while queue:
            node, distance = queue.popleft()
            if node not in visited and distance <= max_distance:
                visited.add(node)
                nodes_within_distance.append(node)
                for edge in graph['edges']:
                    if edge['source'] == node:
                        queue.append((edge['target'], distance + 1))
                    elif edge['target'] == node:
                        queue.append((edge['source'], distance + 1))

        return nodes_within_distance

    def find_edges_between_nodes(self, graph, nodes):
        edges = [edge for edge in graph['edges']
                 if (edge['source'] in nodes and edge['target'] in nodes)]
        return edges

    def is_node_reasonable(self, node, connected_nodes, connected_edges):
        connections_str = ', '.join([f'{e["source"]} {e["type"]} {e["target"]}' for e in connected_edges])
        prompt = (f"Is it reasonable to have a {node} object in this scene? "
                  f"It is connected to the following objects: {', '.join(connected_nodes)}. "
                  f"The connections are as follows: {connections_str}. "
                  f"Answer only with 'reasonable.' or 'not reasonable. Suggest ...'.")
        response = self.llm(prompt)
        if "not reasonable" in response.lower():
            suggestion = response.split("Suggest")[1].strip('.')
            return False, suggestion
        return True, None

    def is_edge_reasonable(self, edge, source_node, target_node):
        prompt = (f"Is it reasonable to have a {source_node} {edge['type']} {target_node} relationship? "
                  f"Answer only with 'reasonable.' or 'not reasonable. Suggest ...'.")
        response = self.llm(prompt)
        if "not reasonable" in response.lower():
            suggestion = response.split("Suggest")[1].strip('.')
            return False, suggestion
        return True, None
    
    def graph_corr(self, nodes_to_check, edges_to_check):
        return nodes_to_check, edges_to_check

    def correct_scene_graph(self, scene_graph, start_node):
        # Step 1: Find nodes within distance
        nodes_to_check = self.find_nodes_within_distance(scene_graph, start_node)

        # Step 2: Find edges between these nodes
        edges_to_check = self.find_edges_between_nodes(scene_graph, nodes_to_check)

        # Step 3 & 4: Check and correct nodes and edges
        corrected_graph = {
            'nodes': [node for node in scene_graph['nodes']],
            'edges': [edge for edge in scene_graph['edges']]
        }

        nodes_to_check, edges_to_check = self.graph_corr(nodes_to_check, edges_to_check)

        # Correct nodes
        for node in nodes_to_check:
            connected_nodes = [n for n in nodes_to_check if n != node]
            connected_edges = [e for e in edges_to_check if e['source'] == node or e['target'] == node]
            is_reasonable, suggestion = self.is_node_reasonable(node, connected_nodes, connected_edges)
            if not is_reasonable:
                if "delete" in suggestion:
                    # Remove the node and all its edges
                    corrected_graph['nodes'] = [n for n in corrected_graph['nodes'] if n['id'] != node]
                    corrected_graph['edges'] = [e for e in corrected_graph['edges'] if e['source'] != node and e['target'] != node]
                else:
                    # Replace the node
                    replacement = suggestion.split()[-1]
                    for n in corrected_graph['nodes']:
                        if n['id'] == node:
                            n['id'] = replacement
                    for e in corrected_graph['edges']:
                        if e['source'] == node:
                            e['source'] = replacement
                        if e['target'] == node:
                            e['target'] = replacement

        # Correct edges
        for edge in edges_to_check:
            is_reasonable, suggestion = self.is_edge_reasonable(edge, edge['source'], edge['target'])
            if not is_reasonable:
                if "delete" in suggestion:
                    # Remove the edge
                    corrected_graph['edges'] = [e for e in corrected_graph['edges'] if e != edge]
                else:
                    # Modify the edge type
                    new_type = suggestion.split()[-1]
                    edge['type'] = new_type

        return corrected_graph
