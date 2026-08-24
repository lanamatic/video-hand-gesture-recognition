"""
Shared evaluation module.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score
)


CLASS_NAMES = [
    "Point-1f", "Point-2f", "Click-1f", "Click-2f",
    "Throw-up", "Throw-down", "Throw-left", "Throw-right",
    "Open-twice", "DblClick-1f", "DblClick-2f", "Zoom-in", "Zoom-out"
]


def full_evaluation(y_true, y_pred, model_name, save_dir="results/figures",
                    y_prob=None, show_plot=True):
    """
    Returns:
        dict with all metrics
    """
    os.makedirs(save_dir, exist_ok=True)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    prec_w = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_w = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    results = {
        'model': model_name,
        'accuracy': acc,
        'f1_weighted': f1_w,
        'f1_macro': f1_m,
        'precision_weighted': prec_w,
        'recall_weighted': rec_w,
    }
    
    # Top-3, Top-5 (if probabilities are available)
    if y_prob is not None:
        y_prob = np.array(y_prob)
        top3 = top_k_accuracy_score(y_true, y_prob, k=3)
        top5 = top_k_accuracy_score(y_true, y_prob, k=5)
        results['top3_accuracy'] = top3
        results['top5_accuracy'] = top5
    
    # Per class report
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        zero_division=0
    )
    results['report'] = report
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    results['confusion_matrix'] = cm
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax
    )
    ax.set_title(f'Confusion Matrix — {model_name}\n'
                 f'Accuracy: {acc:.3f} | F1 (weighted): {f1_w:.3f}',
                 fontsize=14)
    ax.set_ylabel('True label', fontsize=12)
    ax.set_xlabel('Predicted label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"confusion_{model_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    
    # Per class F1 bar plot
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    results['per_class_f1'] = dict(zip(CLASS_NAMES, per_class_f1))
    
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#79B9ED" if f > f1_m else "#FF80F4" for f in per_class_f1]
    ax.bar(CLASS_NAMES, per_class_f1, color=colors, edgecolor='white')
    ax.axhline(y=f1_m, color='gray', linestyle='--', label=f'Macro F1: {f1_m:.3f}')
    ax.set_ylabel('F1-score', fontsize=12)
    ax.set_title(f'Per-class F1-score — {model_name}', fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"per_class_f1_{model_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    
    # Summary
    print(f"\n{'='*50}")
    print(f"  {model_name} — Results")
    print(f"{'='*50}")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  F1 (weighted):     {f1_w:.4f}")
    print(f"  F1 (macro):        {f1_m:.4f}")
    print(f"  Precision (w):     {prec_w:.4f}")
    print(f"  Recall (w):        {rec_w:.4f}")
    if 'top3_accuracy' in results:
        print(f"  Top-3 Accuracy:    {results['top3_accuracy']:.4f}")
        print(f"  Top-5 Accuracy:    {results['top5_accuracy']:.4f}")
    print(f"{'='*50}")
    print(f"\n{report}")
    
    return results


def measure_inference_time(model_predict_fn, X_sample, n_runs=100):
    """
    Measure average inference time per sample.
    Returns:
        dict with mean_ms, std_ms, fps
    """
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model_predict_fn(X_sample)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    times = np.array(times)
    return {
        'mean_ms': times.mean() * 1000,
        'std_ms': times.std() * 1000,
        'fps': 1.0 / times.mean() if times.mean() > 0 else float('inf')
    }


def compare_models(all_results, save_dir="results/figures", show_plot=True):
    """
    Create comparison plots across all models.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model_names = [r['model'] for r in all_results]
    accuracies = [r['accuracy'] for r in all_results]
    f1_weighted = [r['f1_weighted'] for r in all_results]
    f1_macro = [r['f1_macro'] for r in all_results]
    
    # Bar plot: all models side by side 
    x = np.arange(len(model_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width, accuracies, width, label='Accuracy', color="#7BB4E3")
    bars2 = ax.bar(x, f1_weighted, width, label='F1 (weighted)', color="#DFB290")
    bars3 = ax.bar(x + width, f1_macro, width, label='F1 (macro)', color="#F8BAF6")
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    
    # Summary
    print(f"\n{'='*80}")
    print(f"  FINAL MODEL COMPARISON")
    print(f"{'='*80}")
    print(f"  {'Model':<25} {'Accuracy':>10} {'F1 (w)':>10} {'F1 (m)':>10}")
    print(f"  {'-'*55}")
    for r in sorted(all_results, key=lambda x: x['f1_weighted'], reverse=True):
        print(f"  {r['model']:<25} {r['accuracy']:>10.4f} "
              f"{r['f1_weighted']:>10.4f} {r['f1_macro']:>10.4f}")
    print(f"{'='*80}\n")
