import sys
import os
import json
import re
import torch
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chat_models import ChatOllama

#add this line so we can use classes in repo that exist in src folder
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from encoder.EntityEncoder import EntityEncoder
from data_loader import DataLoader

class KGICLLlamaIntegration:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        
        self.entity_encoder = EntityEncoder(args)

        #change pre-trained model depending on the dataset you are using and the dataset you have trained it on
        checkpoint_path = "./checkpoint/pretrain/__['NQ']__train_dim_16_layer_3_rel_layer_2__/model_best.tar"
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{args.gpu}")
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.entity_encoder.load_state_dict(checkpoint['state_dict'], strict=False)
        self.entity_encoder.to(self.device)

        self.llm = ChatOllama(
            model="llama3.1:8b",
            temperature=0.3
        )

        if args.train_dirs is not None:
            self.train_loaders = [DataLoader(args, train_dir, train_dir.split('/')[-2]) 
                                for train_dir in args.train_dirs]
        else:
            self.train_loaders = None
    # Enhanced matching of queries to entities and relations in the knowledge graph using improved similarity search and entity prioritization
    def match_query_to_kg(self, kg_dict, query, top_k=5):

        entity_focus = None
        tell_me_about_pattern = re.compile(r"what (?:can you )?(?:tell|know) (?:me )?about (.*?)\??$", re.IGNORECASE)
        match = tell_me_about_pattern.search(query)
        if match:
            entity_focus = match.group(1).strip()
            print(f"Detected primary entity focus: '{entity_focus}'")
        
        entities_dict = kg_dict.get("entities", {})
        relations_dict = kg_dict.get("relations", {})
        facts_list = kg_dict.get("facts", [])
        
        fact_triples = []
        original_fact_entries = []
        
        for fact_entry in facts_list:
            if isinstance(fact_entry, list) and len(fact_entry) == 1 and isinstance(fact_entry[0], str):
                fact_dict = safe_parse_dict_string(fact_entry[0])
                if fact_dict is None:
                    fact_str = fact_entry[0].replace("'", '"')
                    fact_str = re.sub(r'(\w+)"(\w+)', r'\1\'\2', fact_str)
                    fact_dict = json.loads(fact_str)
                
                for fact_key, sentence_value in fact_dict.items():
                    fact_triples.append(fact_key)
                    original_fact_entries.append(fact_entry)
            else:
                fact_triples.append(fact_entry)
                original_fact_entries.append(fact_entry)
        
        entity_items = list(entities_dict.items())
        relation_items = list(relations_dict.items())
        
        entity_texts = [str(item[1]) for item in entity_items]
        relation_texts = [str(item[1]) for item in relation_items]
        
        stopwords = {'what', 'can', 'you', 'tell', 'me', 'about', 'is', 'are', 'the', 'a', 'an', 'and', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
        
        query_terms = [term.lower() for term in query.split() if term.lower() not in stopwords]
        clean_query = ' '.join(query_terms)
        print(f"Clean query for matching: '{clean_query}'")
        
        entity_vectorizer = TfidfVectorizer(stop_words='english')
        relation_vectorizer = TfidfVectorizer(stop_words='english')
        
        matched_entities = []
        matched_relations = []
        
        # Handle direct entity matches for primary entity focus
        primary_entity_matches = []
        if entity_focus:
            for i, (entity_id, entity_value) in enumerate(entity_items):
                entity_str = str(entity_value).lower()
                if entity_str == entity_focus.lower() or entity_focus.lower() in entity_str:
                    score = 2.0 if entity_str == entity_focus.lower() else 1.5
                    primary_entity_matches.append({
                        "id": entity_id,
                        "value": str(entity_value),
                        "score": score,
                        "is_primary": True
                    })
            
            primary_entity_matches.sort(key=lambda x: x["score"], reverse=True)
            matched_entities.extend(primary_entity_matches)
            print(f"Found {len(primary_entity_matches)} direct matches for primary entity focus")
        
        if entity_texts and clean_query:
            entity_tfidf = entity_vectorizer.fit_transform(entity_texts)
            query_vector = entity_vectorizer.transform([clean_query])
            entity_similarities = cosine_similarity(query_vector, entity_tfidf)[0]
            
            top_entity_indices = entity_similarities.argsort()[::-1][:10]  # Get more candidates
            
            for idx in top_entity_indices:
                if entity_similarities[idx] > 0.1:
                    entity_id = entity_items[idx][0]
                    entity_value = entity_items[idx][1]
                    
                    if any(m["id"] == entity_id for m in primary_entity_matches):
                        continue
                    
                    entity_text = str(entity_value).lower()
                    term_boost = 1.0
                    for term in query_terms:
                        if len(term) > 2 and term.lower() in entity_text:
                            term_boost += 0.2  # Boost for each query term present
                    
                    matched_entities.append({
                        "id": entity_id,
                        "value": str(entity_value),
                        "score": float(entity_similarities[idx]) * term_boost,
                        "is_primary": False
                    })
        
        if relation_texts and clean_query:
            relation_tfidf = relation_vectorizer.fit_transform(relation_texts)
            query_vector = relation_vectorizer.transform([clean_query])
            relation_similarities = cosine_similarity(query_vector, relation_tfidf)[0]
            
            top_relation_indices = relation_similarities.argsort()[::-1][:10]
            
            for idx in top_relation_indices:
                if relation_similarities[idx] > 0.05:
                    relation_id = relation_items[idx][0]
                    relation_value = relation_items[idx][1]
                    
                    relation_text = str(relation_value).lower()
                    term_boost = 1.0
                    for term in query_terms:
                        if len(term) > 2 and term.lower() in relation_text:
                            term_boost += 0.2
                    
                    matched_relations.append({
                        "id": relation_id,
                        "value": str(relation_value),
                        "score": float(relation_similarities[idx]) * term_boost
                    })
        print(f"Found {len(matched_entities)} matching entities and {len(matched_relations)} matching relations")
        
        matched_facts = []
        
        fact_vectorizer = TfidfVectorizer(stop_words='english')
        fact_texts = [str(fact) for fact in fact_triples]
        
        if fact_texts and clean_query:
            try:
                fact_tfidf = fact_vectorizer.fit_transform(fact_texts)
                query_vector = fact_vectorizer.transform([clean_query])
                fact_similarities = cosine_similarity(query_vector, fact_tfidf)[0]
                
                top_fact_indices = fact_similarities.argsort()[::-1][:20]
                
                for idx in top_fact_indices:
                    if fact_similarities[idx] > 0.05:
                        fact_triple = fact_triples[idx]
                        fact_score = float(fact_similarities[idx])
                        
                        term_boost = 1.0
                        fact_str = str(fact_triple).lower()
                        
                        involves_primary = False
                        for entity_match in matched_entities:
                            entity_value = str(entity_match["value"]).lower()
                            if entity_value in fact_str:
                                if entity_match.get("is_primary", False):
                                    fact_score += entity_match["score"] * 3.0
                                    involves_primary = True
                                else:
                                    fact_score += entity_match["score"] * 1.5
                        
                        for rel_match in matched_relations:
                            rel_value = str(rel_match["value"]).lower()
                            if rel_value in fact_str:
                                fact_score += rel_match["score"] * 1.5
                        
                        if entity_focus:
                            entity_focus_lower = entity_focus.lower()
                            if entity_focus_lower in fact_str:
                                fact_score *= 2.0  # Double score for facts mentioning the focus entity
                        
                        original_entry = original_fact_entries[idx]
                        
                        fact_obj = {
                            "score": fact_score,
                            "triple": original_entry,
                        }
                        matched_facts.append(fact_obj)

            except Exception as e:
                print(f"Error in fact matching: {e}")
        
        matched_facts.sort(key=lambda x: (x["score"]), reverse=True)
        return matched_facts[:top_k]

    # Process a user query with a structured prompt for extremely concise answers
    def process_query(self, user_input, loader):
        try:
            try:
                kg_dict = loader.kg.to_dict()
            except (AttributeError, Exception):
                try:
                    kg_dict = {
                        "entities": loader.kg.id2entity if hasattr(loader.kg, 'id2entity') else {},
                        "relations": loader.kg.id2relation if hasattr(loader.kg, 'id2relation') else {},
                        "facts": loader.kg.fact_triple if hasattr(loader.kg, 'fact_triple') else []
                    }
                except (AttributeError, Exception):
                    kg_dict = {
                        "entities": loader.id2entity if hasattr(loader, 'id2entity') else {},
                        "relations": loader.id2relation if hasattr(loader, 'id2relation') else {},
                        "facts": loader.fact_triple if hasattr(loader, 'fact_triple') else []
                    }
            
            matched_facts = self.match_query_to_kg(normalize_kg_data(kg_dict), user_input)
            
            formatted_facts = []
            for fact in matched_facts:
                if isinstance(fact, dict):
                    for key, value in fact.items():
                        if key != 'score':
                            formatted_facts.append(f"{key}: {value}")
            
            facts_text = "\n".join(formatted_facts)
            
            prompt = f"""
            [[ ## system ## ]]
            You are a precise knowledge retrieval system. Your answers must be extremely concise - use at most 10 words.
            If multiple items are requested, format as a comma-separated list. Never use full sentences.
            Always prioritize established, verifiable facts when responding.
            Synthesize provided context with comprehensive domain expertise to ensure factual correctness in every response.
            Do not explain, justify, or add context to your answers.

            [[ ## question ## ]]
            {user_input}

            [[ ## facts ## ]]
            {facts_text}

            [[ ## answer ## ]]
            """
            response = self.llm.invoke(prompt)
            
            answer_text = response.content
            if "[[ ## answer ## ]]" in answer_text:
                answer_text = answer_text.split("[[ ## answer ## ]]")[1].strip()
                if "[[ ##" in answer_text:
                    answer_text = answer_text.split("[[ ##")[0].strip()
            
            return answer_text
            
        except Exception as e:
            print(f"Error in process_query: {e}")
            return "No information available"


def normalize_relation_key(relation_key):
    if isinstance(relation_key, str):
        return relation_key.replace('_', ' ')
    return relation_key

def normalize_kg_data(kg_dict):
    normalized_kg = dict(kg_dict)
    
    if "relations" in normalized_kg:
        for rel_id, rel_value in list(normalized_kg["relations"].items()):
            normalized_kg["relations"][rel_id] = normalize_relation_key(rel_value)
    
    if "facts" in normalized_kg:
        for i, fact in enumerate(normalized_kg["facts"]):
            if isinstance(fact, list) and len(fact) == 3:
                normalized_kg["facts"][i][1] = normalize_relation_key(fact[1])
    return normalized_kg

# Safely parse a string that represents a Python dictionary using regex
def safe_parse_dict_string(dict_str):
    pattern = r"{'([^']*)':\s*'([^']*)'}"
    match = re.match(pattern, dict_str)
    
    if match:
        key = match.group(1)
        value = match.group(2)
        return {key: value}
    else:
        content = dict_str.strip()
        if content.startswith('{') and content.endswith('}'):
            content = content[1:-1]
            
            colon_pos = content.find(':')
            if colon_pos > 0:
                key_str = content[:colon_pos].strip()
                value_str = content[colon_pos + 1:].strip()
                
                if key_str.startswith("'") and key_str.endswith("'"):
                    key_str = key_str[1:-1]
                elif key_str.startswith('"') and key_str.endswith('"'):
                    key_str = key_str[1:-1]
                    
                if value_str.startswith("'") and value_str.endswith("'"):
                    value_str = value_str[1:-1]
                elif value_str.startswith('"') and value_str.endswith('"'):
                    value_str = value_str[1:-1]
                    
                return {key_str: value_str}
    
    return None

def main():
    args = {
        'hidden_dim': 16,
        'attn_dim': 3,
        'n_layer': 3,
        'dropout': 0.0,
        'train_dirs': ['./processed_data/NQ/train'],
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'gpu': 0,
        'act': 'idd',
        'use_prompt_graph': True,
        'use_rspmm': False,
        'use_augment': True,
        'use_token_set': True,
        'relation_mask_rate': 0.5,
        'shot': 3,
        'train_batch_size': 5,
        'test_batch_size': 256,
        'n_relation_encoder_layer': 2,
        'path_hop': 3,
        'MSG': 'concat',
        'AGG': 'max',
        'AGG_rel': 'max',
        'use_attn': True,
        'attn_type': 'Sigmoid',
        'prompt_graph_type': 'all',
        'finetune': False
    }

    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    args = Args(**args)

    print(f"Using device: {args.device}")
    
    integration = KGICLLlamaIntegration(args)

    loader = integration.train_loaders[0] if integration.train_loaders else None
    if loader is None:
        print("No data loader available!")
        return
    
    query = "milo parker starred which 2014 movie alongside gillian anderson?"
    response = integration.process_query(query, loader)
    print(response)

if __name__ == "__main__":
    main()