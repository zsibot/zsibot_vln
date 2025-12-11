class GraphBuilder:
    def __init__(self, llm=None):
        self.llm = llm

    def get_objects(self, llm_response):
        objects = llm_response.split('[')[1].split(']')[0].split(',')
        objects = objects[:5]
        for i in range(len(objects)):
            objects[i] = objects[i].strip()
        return objects

    def get_relations(self, llm_response, objects):
        relations = []
        for line in llm_response.split('\n'):
            relation = {'source': '', 'target': '', 'type': ''}
            parts = line.strip().split(': ')
            if len(parts) == 2:
                relation_info = parts[1].strip()
                relation_parts = relation_info.split(' is ')
                if len(relation_parts) == 2:
                    source_target = parts[0].strip().split(' and ')
                    if len(source_target) == 2:
                        source = source_target[0].strip()
                        target = source_target[1].strip()
                        relation_type = relation_parts[1].strip()
                        if source in objects and target in objects:
                            relation = {'source': source, 'target': target, 'type': relation_type}
                            relations.append(relation)
        return relations

    def parse_text_description(self, description):
        object_prompt = f"""
        Please extract all objects mentioned in the following text description and the output format is "[<object 1>, <object 2>,...]":
        Text Description: {description}
        """

        print("--------object prompt: \n",object_prompt)

        object_response = self.llm(object_prompt)

        print("----LLM response: \n",object_response)
        print("-------------------------------------------------------------------------------------------\n")

        try:
            objects = self.get_objects(object_response)
        except:
            objects = []


        relation_prompt = f"""
        Please describe the relationships between the above objects in the following text description and provide the information in the specified format.
        The format should be: "<Object A> and <Object B>: <Object A> is <relation type> <Object B>".
        If there are multiple relationships, please list them one per line.

        Text Description: {description}

        Example output format:
        Book and Table: Book is on Table
        """

        print("--------relation prompt: \n",relation_prompt)


        relation_response = self.llm(relation_prompt)


        print("----LLM response: \n",relation_response)
        print("-------------------------------------------------------------------------------------------\n")


        relations = self.get_relations(relation_response, objects)

        return objects, relations

    def build_graph(self, objects, relations):
        graph = {
            'nodes': [{'id': obj} for obj in objects],
            'edges': [{'source': r['source'], 'target': r['target'], 'type': r['type']} for r in relations]
        }
        return graph

    def build_graph_from_text(self, text_goal):
        objects, relations = self.parse_text_description(text_goal)
        graph = self.build_graph(objects, relations)
        return graph
