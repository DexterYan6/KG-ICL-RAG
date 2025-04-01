
# A Prompt-based Knowledge Graph Foundation Model for Universal In-Context Reasoning



### Dataset Triple Extraction

The `nq_2_triple` processes the Natural Questions dataset to extract semantic relationships:

```bash
python nq_2_triple.py --input path/to/nq_data.jsonl --output path/to/output_triples.jsonl --model en_core_web_sm
```

The `hotpot_2_triple` file does thie same thing

```bash
python hotpot_2_triple.py --input path/to/nq_data.jsonl --output path/to/output_triples.jsonl --model en_core_web_sm
```

### Triples preprocessor

The `preprocess_tsv` creates the necessary files needed to pretrain the KG-ICL model

```bash
python preprocess.py --dataset hotpot_triples.tsv --output_dir ./processed_data/name of dataset
```
### HotpotQA Evaluation Script
Evaluates KG-ICL (Knowledge Graph Informed Context Learning) on HotpotQA dataset using custom F1, recall, precison, and exact matching.

```bash
python hotpotqa_evaluator.py --hotpotqa_file hotpotqa_data.json
```

### Natural Questions Evaluation Script
Same thing as the HotpotQA eval script but evaluating Natural Questions

```bash
python nq_evaluator.py --nq_file natural_questions.jsonl
```