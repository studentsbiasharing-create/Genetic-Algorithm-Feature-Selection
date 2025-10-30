import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import base64
from io import BytesIO

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

def create_convergence_plot(fitness_history, save_path=None):
    """
    Create convergence plot showing fitness over generations
    
    Parameters:
    -----------
    fitness_history : list
        List of fitness values per generation
    save_path : str, optional
        Path to save the plot
    
    Returns:
    --------
    str: Base64 encoded image or file path
    """
    plt.figure(figsize=(12, 6))
    
    generations = [h['generation'] for h in fitness_history]
    best_fitness = [h['best_fitness'] for h in fitness_history]
    avg_fitness = [h['avg_fitness'] for h in fitness_history]
    worst_fitness = [h['worst_fitness'] for h in fitness_history]
    
    plt.plot(generations, best_fitness, 'g-', linewidth=2, label='Best Fitness', marker='o', markersize=3)
    plt.plot(generations, avg_fitness, 'b--', linewidth=1.5, label='Average Fitness')
    plt.plot(generations, worst_fitness, 'r:', linewidth=1, label='Worst Fitness')
    
    plt.fill_between(generations, worst_fitness, best_fitness, alpha=0.1, color='blue')
    
    plt.xlabel('Generation', fontsize=12, fontweight='bold')
    plt.ylabel('Fitness Score', fontsize=12, fontweight='bold')
    plt.title('Genetic Algorithm Convergence', fontsize=14, fontweight='bold')
    plt.legend(loc='best', frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return f"data:image/png;base64,{image_base64}"

def create_feature_importance_plot(feature_indices, feature_names, scores, method_name, save_path=None, top_n=20):
    """
    Create feature importance bar plot
    
    Parameters:
    -----------
    feature_indices : list
        Indices of features
    feature_names : list
        Names of features
    scores : list
        Importance scores
    method_name : str
        Name of the method
    save_path : str, optional
        Path to save the plot
    top_n : int
        Number of top features to display
    
    Returns:
    --------
    str: Base64 encoded image or file path
    """
    # Get top N features
    top_indices = np.argsort(scores)[::-1][:top_n]
    top_scores = [scores[i] for i in top_indices]
    top_names = [feature_names[i] if i < len(feature_names) else f'Feature {i}' for i in top_indices]
    
    plt.figure(figsize=(12, 8))
    
    colors = sns.color_palette("viridis", len(top_scores))
    bars = plt.barh(range(len(top_scores)), top_scores, color=colors)
    
    plt.yticks(range(len(top_scores)), top_names)
    plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Features - {method_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, top_scores)):
        plt.text(score, i, f' {score:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return f"data:image/png;base64,{image_base64}"

def create_comparison_plot(comparison_results, save_path=None):
    """
    Create comparison bar plot for different methods
    
    Parameters:
    -----------
    comparison_results : dict
        Results from different methods
    save_path : str, optional
        Path to save the plot
    
    Returns:
    --------
    str: Base64 encoded image or file path
    """
    methods = []
    accuracies = []
    n_features = []
    times = []
    
    for key, value in comparison_results.items():
        if 'accuracy' in value:
            method_name = value.get('method', key.upper())
            methods.append(method_name)
            accuracies.append(value['accuracy'] * 100)  # Convert to percentage
            n_features.append(value['n_features'])
            times.append(value.get('execution_time', 0))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Accuracy comparison
    colors = sns.color_palette("Set2", len(methods))
    bars1 = axes[0].bar(range(len(methods)), accuracies, color=colors)
    axes[0].set_xticks(range(len(methods)))
    axes[0].set_xticklabels(methods, rotation=45, ha='right')
    axes[0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Method Comparison - Accuracy', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.2f}%', ha='center', va='bottom', fontsize=9)
    
    # Number of features
    bars2 = axes[1].bar(range(len(methods)), n_features, color=colors)
    axes[1].set_xticks(range(len(methods)))
    axes[1].set_xticklabels(methods, rotation=45, ha='right')
    axes[1].set_ylabel('Number of Features', fontsize=12, fontweight='bold')
    axes[1].set_title('Number of Selected Features', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, nf in zip(bars2, n_features):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{nf}', ha='center', va='bottom', fontsize=9)
    
    # Execution time
    bars3 = axes[2].bar(range(len(methods)), times, color=colors)
    axes[2].set_xticks(range(len(methods)))
    axes[2].set_xticklabels(methods, rotation=45, ha='right')
    axes[2].set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    axes[2].set_title('Execution Time', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, t in zip(bars3, times):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{t:.2f}s', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return f"data:image/png;base64,{image_base64}"

def create_diversity_plot(diversity_history, save_path=None):
    """
    Create diversity plot showing population diversity over generations
    
    Parameters:
    -----------
    diversity_history : list
        List of diversity values per generation
    save_path : str, optional
        Path to save the plot
    
    Returns:
    --------
    str: Base64 encoded image or file path
    """
    plt.figure(figsize=(12, 6))
    
    generations = list(range(len(diversity_history)))
    
    plt.plot(generations, diversity_history, 'b-', linewidth=2, marker='o', markersize=3)
    plt.fill_between(generations, diversity_history, alpha=0.3)
    
    plt.xlabel('Generation', fontsize=12, fontweight='bold')
    plt.ylabel('Population Diversity', fontsize=12, fontweight='bold')
    plt.title('Population Diversity Over Generations', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return f"data:image/png;base64,{image_base64}"

def create_feature_overlap_heatmap(comparison_results, feature_names, save_path=None):
    """
    Create heatmap showing feature overlap between methods
    
    Parameters:
    -----------
    comparison_results : dict
        Results from different methods
    feature_names : list
        Names of all features
    save_path : str, optional
        Path to save the plot
    
    Returns:
    --------
    str: Base64 encoded image or file path
    """
    methods = list(comparison_results.keys())
    n_methods = len(methods)
    
    # Create overlap matrix
    overlap_matrix = np.zeros((n_methods, n_methods))
    
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if 'features' in comparison_results[method1] and 'features' in comparison_results[method2]:
                features1 = set(comparison_results[method1]['features'])
                features2 = set(comparison_results[method2]['features'])
                overlap = len(features1 & features2)
                overlap_matrix[i, j] = overlap
    
    plt.figure(figsize=(10, 8))
    
    method_labels = [comparison_results[m].get('method', m.upper()) for m in methods]
    
    sns.heatmap(overlap_matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                xticklabels=method_labels, yticklabels=method_labels,
                cbar_kws={'label': 'Number of Overlapping Features'})
    
    plt.title('Feature Overlap Between Methods', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return f"data:image/png;base64,{image_base64}"
