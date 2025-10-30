"""
Test script for Genetic Algorithm Feature Selection
"""
import numpy as np
from sklearn.datasets import make_classification
from app.models import GeneticAlgorithm, StatisticalMethods

def test_ga():
    """
    Test genetic algorithm on synthetic dataset
    """
    print("="*60)
    print("Testing Genetic Algorithm Feature Selection")
    print("="*60)
    
    # Generate synthetic dataset
    print("\n1. Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=300,
        n_features=30,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    print(f"   Dataset shape: {X.shape}")
    print(f"   Number of classes: {len(np.unique(y))}")
    
    # Initialize and run GA
    print("\n2. Running Genetic Algorithm...")
    ga = GeneticAlgorithm(
        X, y,
        population_size=30,
        generations=50,
        crossover_rate=0.8,
        mutation_rate=0.1,
        elite_size=3
    )
    
    results = ga.run(verbose=True)
    
    # Run statistical methods
    print("\n3. Running Statistical Methods for Comparison...")
    stat_methods = StatisticalMethods(X, y)
    stat_results = stat_methods.run_all_methods(k=results['n_selected'], verbose=True)
    
    # Compare results
    print("\n4. Comparing with Statistical Methods...")
    comparison = stat_methods.compare_with_ga(results['selected_features'])
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    for method, data in comparison.items():
        if 'accuracy' in data:
            method_name = data.get('method', method.upper())
            print(f"\n{method_name}:")
            print(f"  Features: {data['n_features']}")
            print(f"  Accuracy: {data['accuracy']*100:.2f}%")
            if 'execution_time' in data:
                print(f"  Time: {data['execution_time']:.4f}s")
            if 'overlap_with_ga' in data:
                print(f"  Overlap with GA: {data['overlap_with_ga']} features")
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)

if __name__ == '__main__':
    test_ga()
