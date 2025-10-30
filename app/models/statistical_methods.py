import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    chi2, f_classif, mutual_info_classif,
    SelectKBest, RFE
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
import time

class StatisticalMethods:
    """
    Traditional statistical methods for feature selection
    
    Implements various statistical techniques:
    - Chi-Square Test
    - ANOVA F-Score
    - Mutual Information
    - Correlation Analysis
    - Recursive Feature Elimination (RFE)
    """
    
    def __init__(self, X, y, feature_names=None, random_state=42):
        """
        Initialize Statistical Methods
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        feature_names : list, optional
            Names of features
        random_state : int
            Random seed
        """
        self.X = X
        self.y = y
        self.n_features = X.shape[1]
        self.feature_names = feature_names or [f'Feature_{i}' for i in range(self.n_features)]
        self.random_state = random_state
        self.results = {}
        
    def chi_square_test(self, k='all'):
        """
        Chi-Square test for feature selection
        Measures dependence between features and target
        Works best with non-negative features
        """
        start_time = time.time()
        
        # Ensure non-negative values
        X_scaled = MinMaxScaler().fit_transform(self.X)
        
        if k == 'all':
            k = self.n_features
        
        selector = SelectKBest(chi2, k=k)
        selector.fit(X_scaled, self.y)
        
        scores = selector.scores_
        pvalues = selector.pvalues_
        
        # Rank features
        feature_ranking = np.argsort(scores)[::-1]
        
        execution_time = time.time() - start_time
        
        self.results['chi_square'] = {
            'method': 'Chi-Square Test',
            'scores': scores.tolist(),
            'pvalues': pvalues.tolist(),
            'ranking': feature_ranking.tolist(),
            'top_features': feature_ranking[:k].tolist() if isinstance(k, int) else feature_ranking.tolist(),
            'execution_time': execution_time,
            'feature_names': [self.feature_names[i] for i in feature_ranking[:10]]
        }
        
        return self.results['chi_square']
    
    def anova_f_test(self, k='all'):
        """
        ANOVA F-test for feature selection
        Measures variance between groups
        Good for continuous features
        """
        start_time = time.time()
        
        if k == 'all':
            k = self.n_features
        
        selector = SelectKBest(f_classif, k=k)
        selector.fit(self.X, self.y)
        
        scores = selector.scores_
        pvalues = selector.pvalues_
        
        # Rank features
        feature_ranking = np.argsort(scores)[::-1]
        
        execution_time = time.time() - start_time
        
        self.results['anova_f'] = {
            'method': 'ANOVA F-Test',
            'scores': scores.tolist(),
            'pvalues': pvalues.tolist(),
            'ranking': feature_ranking.tolist(),
            'top_features': feature_ranking[:k].tolist() if isinstance(k, int) else feature_ranking.tolist(),
            'execution_time': execution_time,
            'feature_names': [self.feature_names[i] for i in feature_ranking[:10]]
        }
        
        return self.results['anova_f']
    
    def mutual_information(self, k='all'):
        """
        Mutual Information for feature selection
        Measures mutual dependence between variables
        Captures non-linear relationships
        """
        start_time = time.time()
        
        if k == 'all':
            k = self.n_features
        
        mi_scores = mutual_info_classif(self.X, self.y, 
                                       random_state=self.random_state,
                                       n_neighbors=3)
        
        # Rank features
        feature_ranking = np.argsort(mi_scores)[::-1]
        
        execution_time = time.time() - start_time
        
        self.results['mutual_info'] = {
            'method': 'Mutual Information',
            'scores': mi_scores.tolist(),
            'ranking': feature_ranking.tolist(),
            'top_features': feature_ranking[:k].tolist() if isinstance(k, int) else feature_ranking.tolist(),
            'execution_time': execution_time,
            'feature_names': [self.feature_names[i] for i in feature_ranking[:10]]
        }
        
        return self.results['mutual_info']
    
    def correlation_analysis(self, k='all'):
        """
        Correlation-based feature selection
        Measures linear correlation with target
        """
        start_time = time.time()
        
        if k == 'all':
            k = self.n_features
        
        # Calculate correlation with target
        correlations = []
        for i in range(self.n_features):
            corr = np.corrcoef(self.X[:, i], self.y)[0, 1]
            correlations.append(abs(corr))  # Use absolute value
        
        correlations = np.array(correlations)
        
        # Handle NaN values
        correlations = np.nan_to_num(correlations, nan=0.0)
        
        # Rank features
        feature_ranking = np.argsort(correlations)[::-1]
        
        execution_time = time.time() - start_time
        
        self.results['correlation'] = {
            'method': 'Correlation Analysis',
            'scores': correlations.tolist(),
            'ranking': feature_ranking.tolist(),
            'top_features': feature_ranking[:k].tolist() if isinstance(k, int) else feature_ranking.tolist(),
            'execution_time': execution_time,
            'feature_names': [self.feature_names[i] for i in feature_ranking[:10]]
        }
        
        return self.results['correlation']
    
    def random_forest_importance(self, k='all'):
        """
        Random Forest Feature Importance
        Uses tree-based importance scores
        """
        start_time = time.time()
        
        if k == 'all':
            k = self.n_features
        
        # Optimized: Reduced estimators (100→50) for faster training
        clf = RandomForestClassifier(n_estimators=50, 
                                     random_state=self.random_state,
                                     n_jobs=-1)
        clf.fit(self.X, self.y)
        
        importances = clf.feature_importances_
        
        # Rank features
        feature_ranking = np.argsort(importances)[::-1]
        
        execution_time = time.time() - start_time
        
        self.results['rf_importance'] = {
            'method': 'Random Forest Importance',
            'scores': importances.tolist(),
            'ranking': feature_ranking.tolist(),
            'top_features': feature_ranking[:k].tolist() if isinstance(k, int) else feature_ranking.tolist(),
            'execution_time': execution_time,
            'feature_names': [self.feature_names[i] for i in feature_ranking[:10]]
        }
        
        return self.results['rf_importance']
    
    def recursive_feature_elimination(self, k=10):
        """
        Recursive Feature Elimination (RFE)
        Iteratively removes least important features
        """
        start_time = time.time()
        
        if k == 'all':
            k = self.n_features
        
        # Optimized: Reduced estimators (50\u219230) for faster RFE iterations
        clf = RandomForestClassifier(n_estimators=30, 
                                     random_state=self.random_state,
                                     n_jobs=-1)
        
        rfe = RFE(estimator=clf, n_features_to_select=k, step=1)
        rfe.fit(self.X, self.y)
        
        ranking = rfe.ranking_
        selected_features = np.where(rfe.support_)[0]
        
        execution_time = time.time() - start_time
        
        self.results['rfe'] = {
            'method': 'Recursive Feature Elimination',
            'ranking': ranking.tolist(),
            'selected_features': selected_features.tolist(),
            'top_features': selected_features.tolist(),
            'execution_time': execution_time,
            'feature_names': [self.feature_names[i] for i in selected_features[:10]]
        }
        
        return self.results['rfe']
    
    def run_all_methods(self, k='all', verbose=True):
        """
        Run all statistical methods and compare results
        
        Returns:
        --------
        dict: Results from all methods
        """
        if verbose:
            print("Running Statistical Feature Selection Methods...")
            print("="*60)
        
        methods = [
            ('Chi-Square Test', self.chi_square_test),
            ('ANOVA F-Test', self.anova_f_test),
            ('Mutual Information', self.mutual_information),
            ('Correlation Analysis', self.correlation_analysis),
            ('Random Forest Importance', self.random_forest_importance),
        ]
        
        for method_name, method_func in methods:
            if verbose:
                print(f"\nRunning {method_name}...")
            try:
                result = method_func(k)
                if verbose:
                    print(f"✓ Completed in {result['execution_time']:.4f} seconds")
                    if 'top_features' in result and len(result['top_features']) > 0:
                        print(f"  Top 5 features: {result['top_features'][:5]}")
            except Exception as e:
                if verbose:
                    print(f"✗ Error: {str(e)}")
        
        # Run RFE separately if k is specified
        if k != 'all' and isinstance(k, int):
            if verbose:
                print(f"\nRunning Recursive Feature Elimination...")
            try:
                result = self.recursive_feature_elimination(k)
                if verbose:
                    print(f"✓ Completed in {result['execution_time']:.4f} seconds")
            except Exception as e:
                if verbose:
                    print(f"✗ Error: {str(e)}")
        
        if verbose:
            print("\n" + "="*60)
            print("All methods completed!")
        
        return self.results
    
    def evaluate_feature_subset(self, feature_indices):
        """
        Evaluate performance of a feature subset
        
        Parameters:
        -----------
        feature_indices : list
            Indices of features to evaluate
        
        Returns:
        --------
        float: Cross-validation accuracy
        """
        if len(feature_indices) == 0:
            return 0.0
        
        X_selected = self.X[:, feature_indices]
        
        # Optimized: Reduced estimators (100→50) and CV folds (5→3) for faster evaluation
        clf = RandomForestClassifier(n_estimators=50, 
                                     random_state=self.random_state,
                                     n_jobs=-1)
        
        scores = cross_val_score(clf, X_selected, self.y, cv=3, scoring='accuracy')
        return scores.mean()
    
    def compare_with_ga(self, ga_features):
        """
        Compare statistical methods with GA results
        
        Parameters:
        -----------
        ga_features : list
            Feature indices selected by GA
        
        Returns:
        --------
        dict: Comparison results
        """
        comparison = {
            'ga': {
                'features': ga_features,
                'n_features': len(ga_features),
                'accuracy': self.evaluate_feature_subset(ga_features)
            }
        }
        
        k = len(ga_features)
        
        # Compare each statistical method
        for method_key, method_data in self.results.items():
            if 'top_features' in method_data:
                top_k = method_data['top_features'][:k]
                accuracy = self.evaluate_feature_subset(top_k)
                
                comparison[method_key] = {
                    'method': method_data['method'],
                    'features': top_k,
                    'n_features': len(top_k),
                    'accuracy': accuracy,
                    'execution_time': method_data['execution_time'],
                    'overlap_with_ga': len(set(top_k) & set(ga_features))
                }
        
        return comparison
