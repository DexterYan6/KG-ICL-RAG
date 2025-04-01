
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