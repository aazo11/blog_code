# Small Model Benchmark

This directory contains code and results for benchmarking small language models on various tasks using DSPy. The implementation includes evaluators for GSM8K math problems, PII removal, and prompt rewriting tasks.

## Contents

### Evaluators

- **GSM8K Evaluator** (`dspy_code/gsm8k_eval.py`): Evaluates language models on the GSM8K math word problem dataset.
- **PII Remover Evaluator** (`dspy_code/pii_remover_eval.py`): Evaluates language models on removing Personally Identifiable Information (PII) from text.
- **Prompt Rewriter Evaluator**: Evaluates language models on prompt rewriting tasks.

### Visualization

- **Results Plotter** (`dspy_code/plot_results.py`): A tool to visualize evaluation results, including accuracy and latency metrics across different models.

## Usage

### Running Evaluations

To evaluate a model on GSM8K:
```bash
python3 dspy_code/gsm8k_eval.py --model_name <model_name> --num_questions <number>
```

To evaluate a model on PII removal:
```bash
python3 dspy_code/pii_remover_eval.py --model_name <model_name> --num_examples <number>
```

### Plotting Results

To plot evaluation results:
```bash
python3 dspy_code/plot_results.py --task <task_name>
```

Available tasks:
- `gsm8k`: GSM8K math word problems
- `pii_remover`: PII removal
- `prompt_rewriter`: Prompt rewriting

Options:
- `--no_latency`: Plot only accuracy without latency metrics

## Results

Evaluation results are stored in the `results/` directory, organized by task:
- `results/gsm8k/`: GSM8K evaluation results
- `results/pii_remover/`: PII removal evaluation results
- `results/prompt_rewriter/`: Prompt rewriting evaluation results

Each result file is a JSON containing:
- Model metadata
- Performance metrics (accuracy, latency)
- Detailed results for each example

## Implementation Details

### GSM8K Evaluator
- Uses DSPy for prompt optimization
- Supports both optimized and basic prompts
- Measures accuracy and latency
- Handles model responses with retry logic

### PII Remover Evaluator
- Identifies and removes PII elements from text
- Supports various PII types (names, emails, phone numbers, etc.)
- Evaluates both PII removal accuracy and text readability

### Results Visualization
- Creates bar charts for accuracy and latency
- Color-codes results by model
- Supports filtering by task and metrics 