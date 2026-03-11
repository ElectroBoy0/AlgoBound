# utils.py
import os
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def set_seed(seed):
    """Locks all random number generators for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_and_plot_results(results_list, task_name, model_name, metric_name="MAE"):
    """Saves raw metrics to CSV and generates a multi-line vector PDF plot."""
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # 1. Save CSV
    df = pd.DataFrame(results_list)
    # Using task_name in the CSV so it doesn't overwrite single-model runs
    csv_path = f"results/{task_name}_comparison_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Raw results saved to {csv_path}")
    
    # 2. Plot PDF (Handles multiple models)
    plt.figure(figsize=(8, 5))
    
    # Group the dataframe by model so we get one line per model
    for name, group in df.groupby('model'):
        plt.plot(group['eval_size'], group['metric_value'], marker='o', linewidth=2, label=name)
    
    plt.title(f"OOD Degradation on {task_name.capitalize()}")
    plt.xlabel("Test Dimension (Sequence Length / Node Count)")
    plt.ylabel(f"Error ({metric_name}) - Lower is Better" if metric_name != "Accuracy" else "Accuracy (%) - Higher is Better")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    pdf_path = f"plots/{task_name}_comparison_curve.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Degradation curve saved to {pdf_path}")