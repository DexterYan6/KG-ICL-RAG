import json
import os
import argparse
import spacy
import re
from tqdm import tqdm
from collections import defaultdict
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NQTripleExtractor:
    
    def __init__(self, spacy_model="en_core_web_sm"):
        # Try to load the specified model, fall back to default if needed
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except:
            logger.warning(f"Fallback to default spaCy model after {spacy_model} load failure")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                logger.error("Critical: Language model initialization failed")
                raise Exception("No language processing model available")
    
    def determine_question_type(self, question):
        # Classify questions based on patterns
        q = question.lower()
        
        if re.search(r'^(how many|how much|what is the number|what\'s the number)', q):
            return "count"
        elif re.search(r'^who|whose|which person', q):
            return "person"
        elif re.search(r'^when|what year|what date|which year', q):
            return "time"
        elif re.search(r'^where|which place|which country|which city', q):
            return "location"
        elif re.search(r'^what is|what are|what was|what were', q):
            return "definition"
        elif re.search(r'^which|what', q):
            return "entity"
        else:
            return "other"
    
    def extract_main_entity(self, question):
        # Extract the primary subject from the question
        doc = self.nlp(question)
        
        # Check named entities first
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PERSON", "WORK_OF_ART", "PRODUCT", "EVENT", "FAC"]:
                return ent.text
        
        q_lower = question.lower()
        
        # Handle specific question patterns
        if "how many seasons of" in q_lower:
            match = re.search(r"how many seasons of (.+?)(?:\s+are there|\s+r there|\s+is there|\?|$)", q_lower)
            if match:
                return match.group(1).strip()
        
        if "how many episodes" in q_lower:
            match = re.search(r"how many episodes (?:of|in|are there in) (.+?)(?:\s+are there|\s+r there|\s+is there|\?|$)", q_lower)
            if match:
                return match.group(1).strip()
        
        if "who plays" in q_lower:
            match = re.search(r"who plays (.+?) in (.+?)(?:\?|$)", q_lower)
            if match:
                # Combine context and specific entity
                return match.group(2).strip() + ": " + match.group(1).strip()
        
        # Try noun chunks
        for chunk in doc.noun_chunks:
            if not chunk.text.lower() in ["what", "which", "who", "where", "when", "how", "how many"]:
                if chunk.start > 2:
                    return chunk.text
        
        # Last resort: text after the first verb
        start_idx = 0
        for token in doc:
            if token.pos_ in ["VERB", "AUX"]:
                start_idx = token.i + 1
                break
        
        if start_idx < len(doc) - 1:
            return doc[start_idx:].text
            
        return None
    
    def normalize_answer(self, answer):
        # Standardize answer format 
        answer = answer.strip().rstrip(".,;:!?")
        
        if answer.isdigit():
            return answer
        
        # Convert word numbers to digits
        count_words = {
            "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
            "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
        }
        
        if answer.lower() in count_words:
            return count_words[answer.lower()]
        
        return answer
    
    def create_relation_name(self, question_type, question):
        # Generate relation type based on question
        q_lower = question.lower()
        
        if question_type == "count":
            if "seasons" in q_lower:
                return "number_of_seasons"
            elif "episodes" in q_lower:
                return "number_of_episodes"
            elif "books" in q_lower:
                return "number_of_books"
            elif "films" in q_lower or "movies" in q_lower:
                return "number_of_films"
            else:
                return "count"
                
        elif question_type == "person":
            if "plays" in q_lower:
                return "played_by"
            elif "directed" in q_lower:
                return "directed_by"
            elif "created" in q_lower:
                return "created_by"
            else:
                return "person"
                
        elif question_type == "time":
            if "born" in q_lower:
                return "birth_date"
            elif "died" in q_lower:
                return "death_date"
            elif "founded" in q_lower or "established" in q_lower:
                return "founding_date"
            elif "released" in q_lower:
                return "release_date"
            else:
                return "date"
                
        elif question_type == "location":
            if "born" in q_lower:
                return "birth_place"
            elif "filmed" in q_lower:
                return "filming_location"
            elif "located" in q_lower:
                return "location"
            else:
                return "place"
                
        return question_type
    
    def get_entity_type(self, entity):
        # Determine entity type
        doc = self.nlp(entity)
        
        # Check NER results
        for ent in doc.ents:
            return ent.label_
        
        # Pattern-based fallbacks
        if re.search(r'(movie|film|series|show|episode)', entity.lower()):
            return "WORK_OF_ART"
        elif re.search(r'(album|song|track)', entity.lower()):
            return "WORK_OF_ART"
        elif re.search(r'(company|corporation|inc|llc)', entity.lower()):
            return "ORG"
        elif re.search(r'(game|console)', entity.lower()):
            return "PRODUCT"
        
        return "ENTITY"
    
    def create_additional_triples(self, main_entity, question, answer):
        # Generate extra relationship triples
        triples = []
        
        # Add entity type
        entity_type = self.get_entity_type(main_entity)
        if entity_type:
            triples.append([main_entity, "instance of", entity_type])
        
        # Look for locations
        doc = self.nlp(question + " " + answer)
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                triples.append([main_entity, "narrative location", ent.text])
        
        # Media-specific extractions
        if "how many seasons" in question.lower():
            if "episode" in answer.lower():
                episodes_match = re.search(r'(\d+)\s+episodes', answer.lower())
                if episodes_match:
                    triples.append([main_entity, "number_of_episodes", episodes_match.group(1)])
        
        return triples
    
    def generate_serialized_triples(self, triples):
        # Convert triples to string format
        if not triples:
            return []
            
        serialized = []
        if triples:
            serialized.append(f"{triples[0][0]} {triples[0][1]} {triples[0][2]}")
        
        for triple in triples[1:]:
            serialized.append(f"{triple[1]} {triple[2]}")
            
        return serialized
    
    def process_natural_questions(self, input_file, output_file):
        # Main processing pipeline
        logger.info(f"Reading input file: {input_file}")
        
        data = []
        with open(input_file, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    logger.warning(f"Parsing error on input line")
        
        logger.info(f"Processing {len(data)} question-answer pairs...")
        
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        entity_count = 0
        relation_count = 0
        triple_count = 0
        relation_types = defaultdict(int)
        
        with open(output_file, 'w', encoding='utf-8') as out_file:
            for entry in tqdm(data, desc="Extracting triples"):
                question = entry['question']
                answers = entry['answer']
                
                if not answers or not question:
                    continue
                
                # Analyze question
                question_type = self.determine_question_type(question)
                main_entity = self.extract_main_entity(question)
                
                if not main_entity:
                    continue
                    
                entity_count += 1
                
                relation = self.create_relation_name(question_type, question)
                relation_types[relation] += 1
                relation_count += 1
                
                for answer in answers:
                    normalized_answer = self.normalize_answer(answer)
                    
                    # Build triples
                    main_triple = [main_entity, relation, normalized_answer]
                    additional_triples = self.create_additional_triples(main_entity, question, normalized_answer)
                    
                    all_triples = [main_triple] + additional_triples
                    triple_count += len(all_triples)
                    
                    serialized_triples = self.generate_serialized_triples(all_triples)
                    
                    output = {
                        "triples": all_triples,
                        "serialized_triples": serialized_triples,
                        "sentence": question
                    }
                    
                    out_file.write(json.dumps(output) + "\n")
        
        logger.info(f"Extraction completed. Identified {entity_count} entities, {relation_count} relations.")
        logger.info(f"Generated {triple_count} total triples in {output_file}")
        
        logger.info("Relation type distribution:")
        for rel_type, count in sorted(relation_types.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {rel_type}: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract semantic triples from Natural Questions dataset")
    parser.add_argument("--input", required=True, help="Input Natural Questions JSONL file path")
    parser.add_argument("--output", required=True, help="Output file path for extracted triples")
    parser.add_argument("--model", default="en_core_web_sm", help="SpaCy language model to use")
    
    args = parser.parse_args()
    
    extractor = NQTripleExtractor(spacy_model=args.model)
    extractor.process_natural_questions(args.input, args.output)