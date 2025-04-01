import argparse
import os
import sys
import json
import shutil
import re
import random
from collections import Counter, defaultdict

# Add parent directory to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils_all import *

def read_triples(file_path):
    triples = []
    sentences = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sentence = record.get("sentence", "")
            
            for triple in record.get("triples", []):
                if len(triple) == 3:
                    # Convert to strings and strip whitespace
                    h = str(triple[0]).strip()
                    r = str(triple[1]).strip()
                    t = str(triple[2]).strip()
                    
                    triple_tuple = (h, r, t)
                    triples.append(triple_tuple)
                    sentences[triple_tuple] = sentence
    
    return triples, sentences

def sanitize_filename(name):
    # Replace non-filename chars with underscore
    safe_name = re.sub(r'[^\w\-]', '_', name)
    
    # Avoid file paths that are too long
    if len(safe_name) > 200:
        safe_name = safe_name[:197] + "..."
        
    return safe_name

def count_relation_occurrences(triples):
    relation_counter = Counter()
    for h, r, t in triples:
        relation_counter[r] += 1
    return relation_counter

def filter_triples_by_relation_frequency(triples, relation_counts, min_occurrences=2):
    return [trip for trip in triples if relation_counts[trip[1]] >= min_occurrences]

def sample_triples_by_relation(triples, max_per_relation=500):
    # Group triples by relation
    relation_triples = defaultdict(list)
    for triple in triples:
        relation_triples[triple[1]].append(triple)
    
    sampled_triples = []
    relation_sample_stats = {}
    
    for relation, rel_triples in relation_triples.items():
        if max_per_relation is None or len(rel_triples) <= max_per_relation:
            # Keep all triples for this relation
            sampled_triples.extend(rel_triples)
            relation_sample_stats[relation] = {
                "original_count": len(rel_triples),
                "sampled_count": len(rel_triples),
                "sampling_applied": False
            }
        else:
            # Randomly sample max_per_relation triples
            sampled = random.sample(rel_triples, max_per_relation)
            sampled_triples.extend(sampled)
            relation_sample_stats[relation] = {
                "original_count": len(rel_triples),
                "sampled_count": max_per_relation,
                "sampling_applied": True
            }
    
    return sampled_triples, relation_sample_stats

def split_dataset(triples, train_ratio=0.75, valid_ratio=0.15, test_ratio=0.15):
    # Create a copy of triples to shuffle
    triples_copy = list(set(triples))  # Ensure uniqueness
    random.shuffle(triples_copy)
    
    n = len(triples_copy)
    train_end = int(n * train_ratio)
    valid_end = train_end + int(n * valid_ratio)
    
    splits = {
        'train': triples_copy[:train_end],
        'valid': triples_copy[train_end:valid_end],
        'test': triples_copy[valid_end:]
    }
    
    return splits

def write_sentences(file_path, triples, sentences_dict):
    with open(file_path, 'w', encoding='utf-8') as f:
        for triple in triples:
            sentence = sentences_dict.get(triple, "")
            f.write(f"{sentence}\n")

def write_facts_sentence_mapping(split_dir, triples_ids, entity2id, relation2id, id2entity, id2relation, original_triples, sentences_dict):
    # Read the facts.txt file
    facts = []
    with open(os.path.join(split_dir, 'facts.txt'), 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                facts.append([int(parts[0]), int(parts[1]), int(parts[2])])
    
    # Read the sentences.txt file
    sentences = []
    with open(os.path.join(split_dir, 'sentences.txt'), 'r', encoding='utf-8') as f:
        for line in f:
            sentences.append(line.strip())
    
    # Create mapping file
    with open(os.path.join(split_dir, 'fact_to_sentence.txt'), 'w', encoding='utf-8') as f:
        for i, (h_id, r_id, t_id) in enumerate(facts):
            if i < len(sentences):
                # Get string representations
                h_str = id2entity[h_id] if h_id in id2entity else f"UNKNOWN_ENTITY_{h_id}"
                r_str = id2relation[r_id] if r_id in id2relation else f"UNKNOWN_RELATION_{r_id}"
                t_str = id2entity[t_id] if t_id in id2entity else f"UNKNOWN_ENTITY_{t_id}"
                
                readable_fact = f"{h_str} {r_str} {t_str}"
                sentence = sentences[i]
                
                f.write(f"{{'{readable_fact}': '{sentence}'}}\n")

def process_data(dataset_path, min_relation_occurrences=2, max_relations=None, max_triples_per_relation=500,
                       train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, output_dir=None):
    random.seed(42)

    
    # Check if file exists
    if not os.path.exists(dataset_path):
        print(f"ERROR: Cannot find file {dataset_path}")
        sys.exit(1)
    
    # Read triples
    print(f"Reading triples from {dataset_path}...")
    all_triples, sentences_dict = read_triples(dataset_path)
    
    # Count relation frequencies
    relation_counts = count_relation_occurrences(all_triples)
    total_relations = len(relation_counts)
    
    # Filter out infrequent relations
    filtered_relations = {r: count for r, count in relation_counts.items() 
                         if count >= min_relation_occurrences}
    
    # Further reduce relations if max_relations is specified
    if max_relations and len(filtered_relations) > max_relations:
        print(f"Limiting to top {max_relations} most frequent relations")
        filtered_relations = dict(sorted(filtered_relations.items(), 
                                        key=lambda x: x[1], 
                                        reverse=True)[:max_relations])
    
    relations_to_keep = set(filtered_relations.keys())
    
    print(f"Original relation count: {total_relations}")
    print(f"Relations after filtering (min occurrences={min_relation_occurrences}): {len(filtered_relations)}")
    
    # Filter triples by relation frequency
    filtered_triples = [t for t in all_triples if t[1] in relations_to_keep]
    print(f"Total: {len(all_triples)} -> {len(filtered_triples)} triples after filtering relations")
    
    # Sample triples per relation
    sampled_triples, sample_stats = sample_triples_by_relation(filtered_triples, max_triples_per_relation)
    print(f"Total: {len(filtered_triples)} -> {len(sampled_triples)} triples after sampling (max {max_triples_per_relation} per relation)")
    
    # Split the dataset into train, valid, test
    split_triples = split_dataset(sampled_triples, train_ratio, valid_ratio, test_ratio)
    
    # Create a relation mapping file
    relation_mapping = {}
    
    # Process each split independently
    for split, split_triples in split_triples.items():
        split_dir = os.path.join(output_dir, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir, exist_ok=True)
        
        print(f"Processing {split} split with {len(split_triples)} triples")
        
        # Build entity and relation sets
        entities, relations = get_entities_relations_from_triples(split_triples)
        # Build mapping dictionaries
        entity2id, id2entity = set2dict(entities)
        relation2id, id2relation = set2dict(relations)
        # Create mapping between original relation names and safe filenames
        for relation in relations:
            if relation not in relation_mapping:
                relation_mapping[relation] = sanitize_filename(relation)
        
        # Convert string triples into IDs
        triples_ids = triple2ids(split_triples, entity2id, relation2id)
        
        # Store original triples in order
        ordered_original_triples = []
        for h_id, r_id, t_id in triples_ids:
            h = id2entity[h_id]
            r = id2relation[r_id]
            t = id2entity[t_id]
            ordered_original_triples.append((h, r, t))
        
        # Build Knowledge Graph
        print(f"Building KG graph for {split} split...")
        kg = KG(triples_ids, len(entity2id), len(relation2id))
        
        # Build cases
        cases = kg.build_cases_for_large_graph(case_num=20, enclosing=False, hop=3)
        
        # For training split, sample training data
        if split == 'train':
            train_data = kg.sample_train_data_by_relation(num=200)
            write_triple(os.path.join(split_dir, 'facts.txt'), train_data)

            train_original_triples = []
            for h_id, r_id, t_id in train_data:
                h = id2entity[h_id]
                r = id2relation[r_id]
                t = id2entity[t_id]
                train_original_triples.append((h, r, t))

            write_sentences(os.path.join(split_dir, 'sentences.txt'), train_original_triples, sentences_dict)

            write_facts_sentence_mapping(split_dir, train_data, entity2id, relation2id, id2entity, id2relation,
                                        train_original_triples, sentences_dict)
        else:
            write_triple(os.path.join(split_dir, 'facts.txt'), triples_ids)
            write_sentences(os.path.join(split_dir, 'sentences.txt'), ordered_original_triples, sentences_dict)
            write_facts_sentence_mapping(split_dir, triples_ids, entity2id, relation2id, id2entity, id2relation,
                                        ordered_original_triples, sentences_dict)
        
        # Write common files for each split
        write_triple(os.path.join(split_dir, 'background.txt'), triples_ids)
        write_dict(os.path.join(split_dir, 'entity2id.txt'), entity2id)
        write_dict(os.path.join(split_dir, 'relation2id.txt'), relation2id)
        write_triple(os.path.join(split_dir, 'filter.txt'), triples_ids)
       
        # Write cases for each relation
        cases_dir = os.path.join(split_dir, 'cases')
        os.makedirs(cases_dir, exist_ok=True)
        
        for relation, rel_id in relation2id.items():
            # Use relation name directly for directory
            relation_cases_dir = os.path.join(cases_dir, relation)
            os.makedirs(relation_cases_dir, exist_ok=True)
            
            # Get cases for this relation
            rel_cases = cases.get(rel_id, [])
            if not rel_cases:
                continue
                
            # Write each case
            for i, case in enumerate(rel_cases):
                case_file = os.path.join(relation_cases_dir, str(i))
                write_cases(case_file, case)

    print("Processing complete!")
    print(f"Processed files are available in: {output_dir}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process triples dataset')
    parser.add_argument('--dataset', type=str, default='hotpot_triples.tsv',
                        help='Path to triples file')
    parser.add_argument('--min_occurrences', type=int, default=2,
                        help='Minimum occurrences required to keep a relation')
    parser.add_argument('--max_relations', type=int, default=0,
                        help='Maximum number of relations to keep (None for unlimited)')
    parser.add_argument('--max_triples_per_relation', type=int, default=None,
                        help='Maximum number of triples to keep per relation')
    parser.add_argument('--train_ratio', type=float, default=0.75,
                        help='Proportion of data for training')
    parser.add_argument('--valid_ratio', type=float, default=0.15,
                        help='Proportion of data for validation')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Proportion of data for testing')
    parser.add_argument('--output_dir', type=str, default='./processed_data/hotpot')
    
    args = parser.parse_args()
    
    process_data(
        args.dataset,
        min_relation_occurrences=args.min_occurrences,
        max_relations=args.max_relations,
        max_triples_per_relation=args.max_triples_per_relation,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        output_dir=args.output_dir
    )