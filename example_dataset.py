"""
Generate example datasets for testing feature selection
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, load_iris, load_breast_cancer, load_wine

def generate_synthetic_dataset(n_samples=500, n_features=50, n_informative=15, 
                               n_redundant=10, n_classes=2, random_state=42):
    """
    Generate synthetic classification dataset
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        random_state=random_state,
        shuffle=True
    )
    
    # Create feature names
    feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['Target'] = y
    
    return df

def load_iris_dataset():
    """
    Load Iris dataset
    """
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['Target'] = iris.target
    return df

def load_cancer_dataset():
    """
    Load Breast Cancer dataset
    """
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['Target'] = cancer.target
    return df

def load_wine_dataset():
    """
    Load Wine dataset
    """
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['Target'] = wine.target
    return df

if __name__ == '__main__':
    # Generate and save datasets
    print("Generating example datasets...")
    
    # Synthetic dataset
    df_synthetic = generate_synthetic_dataset()
    df_synthetic.to_csv('uploads/synthetic_dataset.csv', index=False)
    print(f"✓ Synthetic dataset: {df_synthetic.shape[0]} samples, {df_synthetic.shape[1]-1} features")
    
    # Iris dataset
    df_iris = load_iris_dataset()
    df_iris.to_csv('uploads/iris_dataset.csv', index=False)
    print(f"✓ Iris dataset: {df_iris.shape[0]} samples, {df_iris.shape[1]-1} features")
    
    # Breast Cancer dataset
    df_cancer = load_cancer_dataset()
    df_cancer.to_csv('uploads/cancer_dataset.csv', index=False)
    print(f"✓ Cancer dataset: {df_cancer.shape[0]} samples, {df_cancer.shape[1]-1} features")
    
    # Wine dataset
    df_wine = load_wine_dataset()
    df_wine.to_csv('uploads/wine_dataset.csv', index=False)
    print(f"✓ Wine dataset: {df_wine.shape[0]} samples, {df_wine.shape[1]-1} features")
    
    print("\nAll datasets generated successfully!")
