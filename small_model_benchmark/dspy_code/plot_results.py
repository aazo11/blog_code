import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse

def get_all_results(task="gsm8k"):
    """Get all results from the results directory, with model, accuracy, timestamp, and filename."""
    results_dir = os.path.join("results", task)
    os.makedirs(results_dir, exist_ok=True)
    result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    if not result_files:
        print(f"No result files found in {results_dir}")
        return []
    results = []
    for file in result_files:
        filepath = os.path.join(results_dir, file)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            model = data['metadata'].get('model_name', 'unknown')
            accuracy = data['metrics'].get('accuracy', None)
            timestamp = data['metadata'].get('timestamp', None)
            median_latency = data['metrics'].get('median_latency', None)
            results.append({
                'model': model,
                'accuracy': accuracy * 100 if accuracy is not None else None,
                'median_latency': median_latency,
                'timestamp': timestamp,
                'filename': file
            })
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return results

def plot_all_results(task="gsm8k", show_latency=True):
    """Plot all results as bars, colored by model, with a legend. X-axis is run number, bars ordered by accuracy."""
    results = get_all_results(task)
    if not results:
        return
    
    # Create figure with one or two subplots based on show_latency
    if show_latency:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(results)*0.8), 12))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(max(10, len(results)*0.8), 6))
    
    # Sort by accuracy (smallest to largest)
    results = sorted(results, key=lambda x: (x['accuracy'] if x['accuracy'] is not None else -1))
    
    # Prepare data
    run_numbers = list(range(1, len(results) + 1))
    accuracies = [r['accuracy'] for r in results]
    # Filter out None values for latencies
    valid_latency_indices = [i for i, r in enumerate(results) if r['median_latency'] is not None]
    latencies = [results[i]['median_latency'] for i in valid_latency_indices]
    latency_run_numbers = [run_numbers[i] for i in valid_latency_indices]
    models = [r['model'] for r in results]
    unique_models = list(sorted(set(models)))
    
    # Assign a color to each model
    palette = sns.color_palette('tab10', n_colors=len(unique_models))
    model_to_color = {model: palette[i] for i, model in enumerate(unique_models)}
    bar_colors = [model_to_color[m] for m in models]
    latency_bar_colors = [bar_colors[i] for i in valid_latency_indices]
    
    # Plot accuracy
    bars1 = ax1.bar(run_numbers, accuracies, color=bar_colors)
    # Add value labels
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%', 
                ha='center', va='bottom', fontsize=8)
    
    # Plot latency only if show_latency is True and we have valid data
    if show_latency and latencies:
        bars2 = ax2.bar(latency_run_numbers, latencies, color=latency_bar_colors)
        # Add value labels
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}s', 
                    ha='center', va='bottom', fontsize=8)
    
    # Legend
    handles = [plt.Rectangle((0,0),1,1, color=model_to_color[m]) for m in unique_models]
    ax1.legend(handles, unique_models, title="Model")
    
    # Labels for accuracy plot
    task_titles = {
        "gsm8k": "GSM8K",
        "prompt_rewriter": "Prompt Rewriter",
        "pii_remover": "PII Remover"
    }
    task_title = task_titles.get(task, task.title())
    ax1.set_title(f'{task_title} Accuracy for All Runs (Ordered by Accuracy)')
    ax1.set_xlabel('Run Number (ordered by accuracy)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 100)
    ax1.set_xticks(run_numbers)
    
    # Labels for latency plot if showing latency
    if show_latency:
        ax2.set_title(f'{task_title} Median Latency for All Runs')
        ax2.set_xlabel('Run Number (ordered by accuracy)')
        ax2.set_ylabel('Median Latency (seconds)')
        ax2.set_xticks(run_numbers)
    
    plt.tight_layout()
    plt.savefig(f'{task}_results.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{task}_results.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot results from GSM8K, Prompt Rewriter, or PII Remover evaluations.")
    parser.add_argument("--task", type=str, choices=["gsm8k", "prompt_rewriter", "pii_remover"], default="gsm8k",
                      help="Which task's results to plot (default: gsm8k)")
    parser.add_argument("--no_latency", action="store_true",
                      help="Disable latency chart in the plot")
    args = parser.parse_args()
    plot_all_results(args.task, show_latency=not args.no_latency) 