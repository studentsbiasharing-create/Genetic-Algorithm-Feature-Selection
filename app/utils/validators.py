import os
import pandas as pd

ALLOWED_EXTENSIONS = {'csv', 'txt'}

def allowed_file(filename):
    """
    Check if file extension is allowed
    
    Parameters:
    -----------
    filename : str
        Name of the file
    
    Returns:
    --------
    bool: True if extension is allowed
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_dataset(filepath):
    """
    Validate uploaded dataset
    
    Parameters:
    -----------
    filepath : str
        Path to dataset file
    
    Returns:
    --------
    dict: Validation results
    """
    try:
        # Check file exists
        if not os.path.exists(filepath):
            return {
                'valid': False,
                'error': 'File does not exist'
            }
        
        # Check file size
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            return {
                'valid': False,
                'error': 'File is empty'
            }
        
        if file_size > 16 * 1024 * 1024:  # 16MB
            return {
                'valid': False,
                'error': 'File size exceeds 16MB limit'
            }
        
        # Try to read as CSV
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            return {
                'valid': False,
                'error': f'Cannot read CSV file: {str(e)}'
            }
        
        # Validate dataset structure
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
        
        # Check for completely empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            return {
                'valid': False,
                'error': f'Columns {empty_cols} are completely empty'
            }
        
        return {
            'valid': True,
            'n_samples': df.shape[0],
            'n_features': df.shape[1],
            'columns': df.columns.tolist(),
            'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
        }
        
    except Exception as e:
        return {
            'valid': False,
            'error': f'Validation error: {str(e)}'
        }

def validate_ga_parameters(params):
    """
    Validate genetic algorithm parameters
    
    Parameters:
    -----------
    params : dict
        GA parameters
    
    Returns:
    --------
    dict: Validation results
    """
    errors = []
    
    # Population size
    pop_size = params.get('population_size', 50)
    if not isinstance(pop_size, int) or pop_size < 10 or pop_size > 200:
        errors.append('Population size must be between 10 and 200')
    
    # Generations
    generations = params.get('generations', 100)
    if not isinstance(generations, int) or generations < 10 or generations > 500:
        errors.append('Generations must be between 10 and 500')
    
    # Crossover rate
    crossover_rate = params.get('crossover_rate', 0.8)
    if not isinstance(crossover_rate, (int, float)) or crossover_rate < 0 or crossover_rate > 1:
        errors.append('Crossover rate must be between 0 and 1')
    
    # Mutation rate
    mutation_rate = params.get('mutation_rate', 0.1)
    if not isinstance(mutation_rate, (int, float)) or mutation_rate < 0 or mutation_rate > 1:
        errors.append('Mutation rate must be between 0 and 1')
    
    # Elite size
    elite_size = params.get('elite_size', 5)
    if not isinstance(elite_size, int) or elite_size < 1 or elite_size >= pop_size:
        errors.append('Elite size must be between 1 and population_size-1')
    
    if errors:
        return {
            'valid': False,
            'errors': errors
        }
    
    return {
        'valid': True
    }
