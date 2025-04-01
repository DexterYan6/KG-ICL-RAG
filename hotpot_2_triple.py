import spacy
import json
import os
import argparse
import re
from tqdm import tqdm

# Basic logging setup
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TripleExtractor:
    
    def __init__(self, spacy_model="en_core_web_lg"):
        # Try to load the specified model, or fall back to smaller model
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except:
            logger.warning(f"Could not load {spacy_model}, falling back to en_core_web_sm")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                logger.error("Could not load spaCy model. Please install: python -m spacy download en_core_web_sm")
                raise Exception("No spaCy model available")
        
        # Common relation mappings to improve triple quality
        self.relation_mappings = {
            "direct": "director",
            "write": "screenwriter",
            "appear in": "cast member",
            "star in": "cast member",
            "feature": "cast member",
            "be in": "cast member",
            "act in": "cast member",
            "produce": "producer",
            "compose": "composer",
            "edit": "editor",
            "film in": "filming location",
            "set in": "narrative location",
            "take place in": "narrative location",
            "release in": "country of origin",
            "make in": "country of origin",
            "be from": "country of origin"
        }
    
    def normalize_relation(self, relation):
        # Convert relation text to standard format
        relation = relation.lower().strip()
        
        for key, value in self.relation_mappings.items():
            if key in relation:
                return value
                
        # Handle common patterns
        if re.match(r'is an?', relation):
            return "instance of"
        if re.match(r'was an?', relation):
            return "instance of"
        if relation in ["is", "was", "are", "were"]:
            return "instance of"
            
        return relation
    
    def extract_triples_from_sentence(self, sentence, entity_context=None):
        # Parse sentence and extract subject-relation-object triples
        doc = self.nlp(sentence)
        triples = []
        
        # Process with dependency parsing
        for token in doc:
            if token.pos_ in ["VERB", "AUX"]:
                subjects = []
                objects = []
                
                # Add context entity as default subject
                if entity_context:
                    subjects.append(entity_context)
                
                # Find subjects and objects based on dependencies
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subject_text = ' '.join([t.text for t in child.subtree])
                        subjects.append(subject_text.strip())
                    
                    if child.dep_ in ["dobj", "pobj", "attr"]:
                        object_text = ' '.join([t.text for t in child.subtree])
                        objects.append(object_text.strip())
                
                # Create triples from subjects and objects
                for subject in subjects:
                    for obj in objects:
                        if subject and obj and subject.lower() != obj.lower():
                            relation = self.normalize_relation(token.lemma_)
                            triples.append([subject, relation, obj])
        
        # If no triples were found and there's an entity context, try extracting entities
        if not triples and entity_context:
            # Add "instance of" relation for recognized entities
            for ent in doc.ents:
                if ent.label_ in ["WORK_OF_ART", "ORG", "PRODUCT", "EVENT"]:
                    triples.append([entity_context, "instance of", ent.label_])
                    
                # Add entity-specific relations
                if ent.label_ == "PERSON":
                    triples.append([entity_context, "cast member", ent.text])
                elif ent.label_ == "GPE" or ent.label_ == "LOC":
                    triples.append([entity_context, "narrative location", ent.text])
        
        # Add film-specific patterns
        if "film" in sentence.lower() or "movie" in sentence.lower():
            # Check for director pattern
            director_match = re.search(r"directed by\s+([^,.]+)", sentence, re.IGNORECASE)
            if director_match and entity_context:
                triples.append([entity_context, "director", director_match.group(1).strip()])
            
            # Check for cast pattern
            if entity_context:
                # Look for lists of actors
                if "stars" in sentence.lower() or "starring" in sentence.lower() or "cast" in sentence.lower():
                    # Extract names from lists
                    names = re.findall(r"([A-Z][a-z]+ [A-Z][a-z]+)", sentence)
                    for name in names:
                        triples.append([entity_context, "cast member", name])
            
            # Add film instance
            if entity_context and "film" in entity_context.lower() and not any(t[1] == "instance of" for t in triples):
                triples.append([entity_context, "instance of", "Film"])
        
        # Add country pattern
        country_names = ["United States", "United Kingdom", "France", "Germany", "Italy", 
                         "Spain", "Canada", "Australia", "Japan", "China", "India", 
                         "Russia", "Brazil", "Mexico", "Argentina", "Chile", "Peru"]
        
        if entity_context:
            for country in country_names:
                if country in sentence:
                    triples.append([entity_context, "country of origin", country])
        
        return triples, sentence
    
    def generate_serialized_triples(self, triples):
        # Convert triples to text format
        if not triples:
            return []
            
        serialized = []
        # First triple has full subject
        if triples:
            serialized.append(f"{triples[0][0]} {triples[0][1]} {triples[0][2]}")
        
        # Subsequent triples omit repeated subject
        for triple in triples[1:]:
            serialized.append(f"{triple[1]} {triple[2]}")
            
        return serialized
    
    def process_hotpotqa(self, input_file, output_file):
        # Main method to process dataset and extract triples
        logger.info(f"Reading input file: {input_file}")
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Processing {len(data)} documents...")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as out_file:
            for entry in tqdm(data, desc="Processing documents"):
                # Process context passages
                for title, sentences in entry['context']:
                    entity_context = title
                    
                    # Process each sentence separately
                    for sentence in sentences:
                        if sentence.strip():
                            triples, sent = self.extract_triples_from_sentence(sentence, entity_context)
                            
                            # Only output if triples were found
                            if triples:
                                serialized_triples = self.generate_serialized_triples(triples)
                                
                                # Create output in the specified format
                                output = {
                                    "triples": triples,
                                    "serialized_triples": serialized_triples,
                                    "sentence": sent
                                }
                                
                                # Write as JSON line
                                out_file.write(json.dumps(output) + "\n")
                
                # Process supporting facts
                for title, sent_id in entry.get('supporting_facts', []):
                    for context_title, context_sentences in entry['context']:
                        if context_title == title and sent_id < len(context_sentences):
                            sentence = context_sentences[sent_id]
                            if sentence.strip():
                                triples, sent = self.extract_triples_from_sentence(sentence, title)
                                
                                # Only output if triples were found
                                if triples:
                                    serialized_triples = self.generate_serialized_triples(triples)
                                    
                                    # Create output in the specified format
                                    output = {
                                        "triples": triples,
                                        "serialized_triples": serialized_triples,
                                        "sentence": sent
                                    }
                                    
                                    # Write as JSON line
                                    out_file.write(json.dumps(output) + "\n")
        
        logger.info(f"Triples extracted and saved to {output_file}")

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract triples from HotpotQA in specified format")
    parser.add_argument("--input", required=True, help="Input HotpotQA JSON file")
    parser.add_argument("--output", required=True, help="Output file for triples in JSON lines format")
    parser.add_argument("--model", default="en_core_web_lg", help="SpaCy model to use")
    
    args = parser.parse_args()
    
    extractor = TripleExtractor(spacy_model=args.model)
    extractor.process_hotpotqa(args.input, args.output)