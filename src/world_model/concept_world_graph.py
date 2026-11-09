class ConceptWorldGraph:

    def __init__(self):
        self.graph = {}

    def add_relation(
        self,
        parent,
        child,
        relation="contains"
    ):
        if parent not in self.graph:
            self.graph[parent] = []

        self.graph[parent].append({
            "child": child,
            "relation": relation
        })

    def get_children(
        self,
        parent
    ):
        return self.graph.get(
            parent,
            []
        )

    def get_graph(self):
        return self.graph