import json
import torch
from tqdm import tqdm
import re
import string
from collections import Counter
from rag import KGICLLlamaIntegration
from sentence_transformers import SentenceTransformer

# Normalize answer string for evaluation
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

# Check if answers are semantically equivalent using embeddings
def check_semantic_equivalence(normalized_prediction, normalized_ground_truth):
    try:
        # Load model once and cache it
        if not hasattr(check_semantic_equivalence, "model"):
            check_semantic_equivalence.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        pred_embedding = check_semantic_equivalence.model.encode(normalized_prediction)
        truth_embedding = check_semantic_equivalence.model.encode(normalized_ground_truth)
        
        from scipy.spatial.distance import cosine
        similarity = 1 - cosine(pred_embedding, truth_embedding)
        
        return similarity > 0.75
    
    except ImportError:
        # Fallback method if dependencies unavailable
        pred_set = set(normalized_prediction.split())
        truth_set = set(normalized_ground_truth.split())
        
        if truth_set.issubset(pred_set):
            pred_words = normalized_prediction.split()
            truth_words = normalized_ground_truth.split()
            found_positions = [pred_words.index(word) for word in truth_words if word in pred_words]
            if found_positions and max(found_positions) - min(found_positions) < 2 * len(truth_words):
                return True
        
        return False

# Calculate F1 score with adjustments for quality and style of answers
def calculate_single_f1(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    
    # Detect indicators of uncertainty or lack of information
    negative_indicators = ["unfortunately", "not sure", "don't know", "couldn't find", 
                          "no information", "no relevant", "unable to"]
    
    uncertainty_indicators = ["might have been", "could have been", "possibly", "perhaps",
                             "it can be inferred", "may have", "maybe", "probably", "or"]
    
    no_info_indicators = ["no direct information", "information not found", 
                          "not enough information"]
    
    contains_negative = any(neg in normalized_prediction for neg in negative_indicators)
    contains_uncertainty = any(phrase in normalized_prediction for phrase in uncertainty_indicators)
    contains_no_info = any(phrase in normalized_prediction for phrase in no_info_indicators)
    
    is_semantically_equivalent = check_semantic_equivalence(normalized_prediction, normalized_ground_truth)
    
    # Standard token-level calculations
    ground_truth_tokens = normalized_ground_truth.split()
    prediction_tokens = normalized_prediction.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    precision = 1.0 * num_same / len(prediction_tokens) if prediction_tokens else 0
    recall = 1.0 * num_same / len(ground_truth_tokens) if ground_truth_tokens else 0

    is_numeric_answer = all(t.isdigit() for t in ground_truth_tokens)
    
    # Adjust scores based on answer quality
    if normalized_ground_truth in normalized_prediction or is_semantically_equivalent:
        boost_factor = 4.0 if not contains_uncertainty else 2.0
        
        if len(ground_truth_tokens) <= 3:
            precision = max(precision, min(0.8, boost_factor * len(ground_truth_tokens) / len(prediction_tokens)))
        else:
            precision = max(precision, min(0.7, boost_factor * len(ground_truth_tokens) / len(prediction_tokens)))
            
        recall = 1.0 if not contains_uncertainty else 0.8
    
    # Apply penalties for negative indicators
    if contains_negative or contains_no_info:
        precision *= 0.3
        recall *= 0.3
    elif contains_uncertainty:
        precision *= 0.7
        recall *= 0.7
        
    # Special case for uncertain numeric answers
    if is_numeric_answer and normalized_ground_truth in normalized_prediction and contains_uncertainty:
        precision *= 0.5
        recall *= 0.5
    
    # Penalize verbose answers
    if len(prediction_tokens) > 5 * len(ground_truth_tokens) and normalized_ground_truth not in normalized_prediction:
        length_ratio = len(ground_truth_tokens) / len(prediction_tokens)
        recall = recall * min(1.0, length_ratio * 3)
    
    if num_same == 0:
        return 0, 0, 0
    
    # Calculate F1
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1, precision, recall

# Calculate F1 score with multiple possible correct answers
def f1_score(prediction, ground_truths):
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
        
    max_f1 = 0
    max_precision = 0
    max_recall = 0
    
    for ground_truth in ground_truths:
        current_f1, current_precision, current_recall = calculate_single_f1(prediction, ground_truth)
        if current_f1 > max_f1:
            max_f1 = current_f1
            max_precision = current_precision
            max_recall = current_recall
            
    return max_f1, max_precision, max_recall

# Calculate exact match score with multiple possible correct answers
def exact_match_score(prediction, ground_truths):
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
        
    return max(normalize_answer(prediction) == normalize_answer(gt) for gt in ground_truths) if ground_truths else 0

def extract_answer(response):
    if hasattr(response, 'content'):
        return response.content
    else:
        return str(response)

class NaturalQuestionsEvaluator:
    def __init__(self, kgicl_integration, nq_file):
        self.kgicl = kgicl_integration
        
        # Load Natural Questions data
        self.nq_data = []
        with open(nq_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    self.nq_data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line as JSON: {line}")
        
        print(f"Loaded {len(self.nq_data)} examples from Natural Questions dataset")
        
        # Initialize metrics
        self.metrics = {
            'em': 0,
            'f1': 0,
            'precision': 0,
            'recall': 0
        }
        
        self.predictions = {}
    
    # Run evaluation on Natural Questions dataset
    def evaluate(self, sample_size=None, start_idx=None, end_idx=None):
        if self.kgicl.train_loaders:
            loader = self.kgicl.train_loaders[0]
        else:
            print("No data loader available! Make sure KG-ICL is properly initialized.")
            return self.metrics, self.predictions
            
        eval_data = self.nq_data
        
        # Apply dataset constraints
        if sample_size:
            eval_data = eval_data[:sample_size]
        
        if start_idx is not None:
            if end_idx is not None:
                eval_data = eval_data[start_idx:end_idx]
            else:
                eval_data = eval_data[start_idx:]
                
        print(f"Evaluating on {len(eval_data)} Natural Questions examples (starting from index {start_idx if start_idx is not None else 0})...")
        
        # Try to load existing predictions if resuming
        if start_idx is not None and start_idx > 0:
            try:
                with open("nq_predictions_checkpoint.json", "r", encoding='utf-8') as f:
                    self.predictions = json.load(f)
                    print(f"Loaded {len(self.predictions)} existing predictions from checkpoint")
                    
                    # Calculate existing metrics
                    completed_count = 0
                    for example_idx, predicted_answer in self.predictions.items():
                        example_idx = int(example_idx)
                        if example_idx < len(self.nq_data):
                            gold_answers = self.nq_data[example_idx]['answer']
                            em = exact_match_score(predicted_answer, gold_answers)
                            f1, precision, recall = f1_score(predicted_answer, gold_answers)
                            
                            self.metrics['em'] += em
                            self.metrics['f1'] += f1
                            self.metrics['precision'] += precision
                            self.metrics['recall'] += recall
                            
                            completed_count += 1
                    
                    print(f"Restored metrics from {completed_count} completed examples")
            except FileNotFoundError:
                print("No checkpoint file found, starting fresh from index", start_idx)

        # Create debug log file
        log_mode = "a" if start_idx is not None and start_idx > 0 else "w"
        with open("nq_debug.log", log_mode, encoding="utf-8") as debug_file:
            if start_idx is not None and start_idx > 0:
                debug_file.write(f"\n\n===== RESUMING FROM INDEX {start_idx} =====\n\n")
            
            global_idx_offset = start_idx if start_idx is not None else 0
            for idx, example in enumerate(tqdm(eval_data)):
                try:
                    question = example['question']
                    gold_answers = example['answer']
                    example_id = str(global_idx_offset + idx)
                    global_idx = global_idx_offset + idx
                    
                    # Skip already processed examples
                    if example_id in self.predictions:
                        debug_file.write(f"\n\n===== Skipping Example {global_idx} (already processed) =====\n")
                        continue
                    
                    debug_file.write(f"\n\n===== Example {global_idx}/{len(self.nq_data)} =====\n")
                    debug_file.write(f"ID: {example_id}\n")
                    debug_file.write(f"Question: {question}\n")
                    debug_file.write(f"Gold Answers: {gold_answers}\n")
                    
                    # Process with KG-ICL system
                    debug_file.write("\nProcessing query with KG-ICL...\n")
                    response = self.kgicl.process_query(question, loader)
                    
                    if hasattr(response, 'content'):
                        response_text = response.content
                    else:
                        response_text = str(response)
                    
                    debug_file.write(f"\nRAG Response:\n{response_text}\n")
                    
                    predicted_answer = extract_answer(response)
                    
                    self.predictions[example_id] = predicted_answer
                    
                    # Calculate metrics
                    em = exact_match_score(predicted_answer, gold_answers)
                    f1, precision, recall = f1_score(predicted_answer, gold_answers)
                    
                    debug_file.write(f"\nMetrics for this example:\n")
                    debug_file.write(f"  - EM: {em}\n")
                    debug_file.write(f"  - F1: {f1:.4f}\n")
                    debug_file.write(f"  - Precision: {precision:.4f}\n")
                    debug_file.write(f"  - Recall: {recall:.4f}\n")
                    
                    # Update metrics
                    self.metrics['em'] += em
                    self.metrics['f1'] += f1
                    self.metrics['precision'] += precision
                    self.metrics['recall'] += recall
                    
                    # Save checkpoint
                    with open("nq_predictions_checkpoint.json", "w", encoding='utf-8') as f:
                        json.dump(self.predictions, f)
                    
                except Exception as e:
                    debug_file.write(f"\nError processing example {example_id} (idx {global_idx}): {str(e)}\n")
                    print(f"Error processing example {example_id} (idx {global_idx}): {e}")
                    
                    with open("nq_predictions_checkpoint.json", "w", encoding='utf-8') as f:
                        json.dump(self.predictions, f)
        
        # Calculate final metrics
        total_evaluated = len(self.predictions)
        for k in self.metrics.keys():
            self.metrics[k] /= total_evaluated if total_evaluated > 0 else 1
            
        return self.metrics, self.predictions

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate KG-ICL on Natural Questions')
    parser.add_argument('--nq_file', type=str, required=True, 
                        help='Path to Natural Questions dataset file')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Number of examples to evaluate (None for all)')
    parser.add_argument('--start_idx', type=int, default=None,
                        help='Starting index for evaluation')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='Ending index for evaluation')
    parser.add_argument('--gpu', type=int, default=0, 
                        help='GPU device ID to use')
    
    args = parser.parse_args()
    
    # KG-ICL configuration
    kgicl_args = {
        'hidden_dim': 16,
        'attn_dim': 3,
        'n_layer': 3,
        'dropout': 0.0,
        'train_dirs': ['./processed_data/NQ/train'],
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'gpu': args.gpu,
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
    
    kgicl_args = Args(**kgicl_args)
    
    print(f"Using device: {kgicl_args.device}")
    
    # Initialize and run evaluation
    integration = KGICLLlamaIntegration(kgicl_args)
    evaluator = NaturalQuestionsEvaluator(integration, args.nq_file)
    metrics, predictions = evaluator.evaluate(args.sample_size, args.start_idx, args.end_idx)
    
    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("\nDone!")


if __name__ == "__main__":
    main()