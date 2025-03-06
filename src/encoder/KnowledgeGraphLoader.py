import os

class KnowledgeGraphLoader:
    def __init__(self, kg_data_path="../../checkpoint", device='cpu'):
        """
        Initializes the Knowledge Graph loader with the path to the KG data stored in `checkpoint/`.
        Args:
            kg_data_path: Path to the knowledge graph data.
            device: Device to store data on, default is CPU.
        """
        self.kg_data_path = kg_data_path
        self.device = device
        self.entity_num = 0  # To be set after loading the data
        self.relation_num = 0  # To be set after loading the data
        self.entities = []  # List of entities in the knowledge graph
        self.relations = []  # List of relations in the knowledge graph
        self.triples = []  # List of triples (subject, relation, object)
        self.load_kg()  # Load the knowledge graph

    def load_kg(self):
        """
        Loads the knowledge graph from the checkpoint directory.
        Assumes that the knowledge graph triples are stored in a file (e.g., `kg_triples.txt`).
        """
        if not os.path.exists(self.kg_data_path):
            raise FileNotFoundError(f"Knowledge graph file not found at {self.kg_data_path}")
        
        with open(self.kg_data_path, 'r') as file:
            for line in file:
                # Assuming each line is a triple in the format: subject, relation, object
                sub, rel, obj = line.strip().split('\t')  # Adjust split if a different delimiter is used
                self.triples.append((sub, rel, obj))
                
                # Add unique entities and relations
                if sub not in self.entities:
                    self.entities.append(sub)
                if obj not in self.entities:
                    self.entities.append(obj)
                if rel not in self.relations:
                    self.relations.append(rel)

        self.entity_num = len(self.entities)
        self.relation_num = len(self.relations)
        print(f"Loaded {len(self.triples)} triples, {self.entity_num} entities, {self.relation_num} relations.")

    def get_triples_for_entity(self, entity):
        """
        Retrieves all triples related to a given entity.
        Args:
            entity: The entity for which we need to retrieve triples.
        """
        related_triples = []
        for sub, rel, obj in self.triples:
            if sub == entity or obj == entity:
                related_triples.append((sub, rel, obj))
        return related_triples

    def get_neighbors(self, entity):
        """
        Retrieves neighbors (subject-object pairs) for a given entity.
        Args:
            entity: The entity for which neighbors are to be retrieved.
        """
        neighbors = []
        for sub, rel, obj in self.triples:
            if sub == entity:
                neighbors.append((obj, rel))  # Object of the subject
            if obj == entity:
                neighbors.append((sub, rel))  # Subject of the object
        return neighbors