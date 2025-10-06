# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments
```bash
# Run main inference pipeline on all samples
python main.py

# Run on specific number of samples (e.g., 100)
python main.py -n 100

# Run with debug mode
python main.py -n 10 -d

# Run RQ1 analysis
python run_rq1.py
```

### Code Quality
```bash
# Code formatting
black *.py

# Code linting
pylint *.py

# Run tests
pytest
```

## Architecture Overview

### Core Components

1. **main.py** - Main inference pipeline that:
   - Loads Llama-2-7b-chat-hf model
   - Processes MetaQA-1hop samples with trimmed subgraphs
   - Calculates TUS (Triple Utilization Score), GASS, and GASS-JSD metrics
   - Evaluates using Hit@1 accuracy

2. **Metrics Calculation Modules**:
   - `tus_metrics.py` - Triple Utilization Score calculation
   - `tus_variants.py` - Multiple TUS variants (strict, contrast, relative, etc.)
   - `gass_metrics.py` - Graph-based Alignment Score with Subgraph
   - `gass_jsd_metrics.py` - GASS with Jensen-Shannon Divergence

3. **Data Processing**:
   - `dataset_processor.py` - Loads MetaQA data, entities, relations
   - `subgraph_utils.py` - Subgraph selection and trimming algorithms
   - `gold_expansion_utils.py` - Gold expansion set processing

4. **Evaluation**:
   - `squad_style_evaluator.py` - SQuAD-style evaluation metrics
   - `metrics_utils.py` - General evaluation utilities

### Data Flow

1. **Input**: MetaQA-1hop dataset with question-specific subgraphs
2. **Preprocessing**: Subgraph trimming to top-20 most relevant triples using shortest path + importance scoring
3. **Inference**: Feed trimmed subgraph + question to Llama model
4. **Evaluation**: Calculate Hit@1 accuracy and analyze attention patterns via TUS/GASS metrics
5. **Output**: Results saved as timestamped JSONL files in `experiment_records/`

### Key Algorithms

**Subgraph Trimming Strategy**:
- Find shortest paths from question entities to answer entities
- Score neighboring triples by connectivity (3 points for path connections, 2 for question/answer entity connections)
- Select top-K triples maintaining path completeness
- Fallback to answer-containing triples if path finding fails

**Metrics Design**:
- **TUS**: Measures attention alignment between model and external knowledge triples
- **GASS**: Evaluates model's utilization of gold vs retrieved subgraph information
- **Hit@1**: Standard accuracy metric for first prediction correctness

## File Structure Context

- `experiment_records/` - All experimental results and model checkpoints
- `dataset/metaqa-1hop/` - MetaQA dataset files (entities.txt, relations.txt)
- Analysis scripts in `analysis/` directory for result processing
- Hallucination detection models stored in `experiment_records/detector/`

## Data Formats

**Trimmed Results Format** (input to main.py):
- Each line: JSON with question, trimmed_triples, gold_triples, gold_expansion_set
- Triple format: [head_entity_id, relation_id, tail_entity_id]
- Entity IDs reference line numbers in entities.txt (Freebase IDs)
- Relation IDs reference line numbers in relations.txt (Freebase relations)

**Inference Results Format** (output from main.py):
- First line: Configuration metadata
- Middle lines: Per-sample results with scores and predictions
- Last line: Aggregate statistics

## Model Configuration

- **Model**: meta-llama/Llama-2-7b-chat-hf
- **Device**: Auto-detection with GPU preference
- **Batch Size**: 2 (configurable for memory management)
- **Generation**: Max 50 new tokens, greedy decoding
- **Precision**: FP16 on GPU, FP32 on CPU