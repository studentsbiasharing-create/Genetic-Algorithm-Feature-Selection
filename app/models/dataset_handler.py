import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

class DatasetHandler:
    """
    Handle dataset loading, preprocessing, and validation
    """
    
    def __init__(self, filepath, target_column=None):
        """
        Initialize Dataset Handler
        
        Parameters:
        -----------
        filepath : str
            Path to CSV file
        target_column : str or int, optional
            Name or index of target column (default: last column)
        """
        self.filepath = filepath
        self.target_column = target_column
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.target_name = None
        self.label_encoder = None
        self.scaler = None
        self.dataset_info = {}
        
    def load_data(self):
        """
        Load dataset from CSV file
        """
        try:
            self.df = pd.read_csv(self.filepath)
            return True
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
    
    def preprocess(self, scale_features=True, handle_missing=True):
        """
        Preprocess the dataset
        
        Parameters:
        -----------
        scale_features : bool
            Whether to standardize features
        handle_missing : bool
            Whether to handle missing values
        """
        if self.df is None:
            raise Exception("Dataset not loaded. Call load_data() first.")
        
        # Identify target column
        if self.target_column is None:
            self.target_column = self.df.columns[-1]
        elif isinstance(self.target_column, int):
            self.target_column = self.df.columns[self.target_column]
        
        self.target_name = self.target_column
        
        # Drop rows with missing target values to avoid encoding issues
        if self.df[self.target_column].isna().any():
            self.df = self.df.dropna(subset=[self.target_column]).reset_index(drop=True)
        
        # Separate features and target
        X_df = self.df.drop(columns=[self.target_column])
        y_series = self.df[self.target_column]
        
        self.feature_names = X_df.columns.tolist()
        
        # If there are non-numeric columns, encode them
        non_numeric_cols = X_df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            for col in non_numeric_cols:
                le = LabelEncoder()
                X_df[col] = le.fit_transform(X_df[col].astype(str))
        
        # Coerce all features to numeric; non-convertible values become NaN
        X_df = X_df.apply(pd.to_numeric, errors='coerce')
        
        # Ensure numpy float array for downstream operations
        self.X = X_df.values.astype(np.float64, copy=False)
        
        # Handle missing values (always impute when enabled to avoid isnan on object dtypes)
        if handle_missing:
            imputer = SimpleImputer(strategy='mean')
            self.X = imputer.fit_transform(self.X)
        
        # Encode target if categorical
        if y_series.dtype == 'object' or y_series.dtype.name == 'category':
            self.label_encoder = LabelEncoder()
            self.y = self.label_encoder.fit_transform(y_series)
        else:
            self.y = y_series.values
        
        # Scale features
        if scale_features:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
        
        # Collect dataset information
        self.dataset_info = {
            'n_samples': self.X.shape[0],
            'n_features': self.X.shape[1],
            'n_classes': len(np.unique(self.y)),
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'class_distribution': {
                str(k): int(v) for k, v in zip(*np.unique(self.y, return_counts=True))
            },
            'missing_values': int(pd.isna(self.df).to_numpy().sum()),
            'feature_stats': self._get_feature_stats()
        }
        
        return self.X, self.y
    
    def _get_feature_stats(self):
        """
        Get statistical summary of features
        """
        stats = []
        for i, name in enumerate(self.feature_names):
            feature_data = self.X[:, i]
            stats.append({
                'name': name,
                'mean': float(np.mean(feature_data)),
                'std': float(np.std(feature_data)),
                'min': float(np.min(feature_data)),
                'max': float(np.max(feature_data)),
                'median': float(np.median(feature_data))
            })
        return stats
    
    def get_info(self):
        """
        Get dataset information
        """
        return self.dataset_info
    
    def get_feature_names(self):
        """
        Get list of feature names
        """
        return self.feature_names
    
    def get_data(self):
        """
        Get preprocessed features and target
        """
        return self.X, self.y
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into train and test sets
        """
        return train_test_split(self.X, self.y, 
                               test_size=test_size, 
                               random_state=random_state,
                               stratify=self.y)
    
    @staticmethod
    def validate_dataset(filepath):
        """
        Validate dataset file
        
        Returns:
        --------
        dict: Validation results
        """
        try:
            df = pd.read_csv(filepath)
            
            if df.shape[0] < 10:
                return {
                    'valid': False,
                    'error': 'Dataset must have at least 10 samples'
                }
            
            if df.shape[1] < 2:
                return {
                    'valid': False,
                    'error': 'Dataset must have at least 2 columns (features + target)'
                }
            
            return {
                'valid': True,
                'n_samples': df.shape[0],
                'n_features': df.shape[1],
                'columns': df.columns.tolist()
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
