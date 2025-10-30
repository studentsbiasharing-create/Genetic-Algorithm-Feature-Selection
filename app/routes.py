from flask import Blueprint, render_template, request, jsonify, current_app, send_from_directory
import os
import json
import time
from werkzeug.utils import secure_filename
from app.models import GeneticAlgorithm, StatisticalMethods, DatasetHandler
from app.utils import (
    allowed_file, validate_dataset, 
    create_comparison_plot, create_convergence_plot, 
    create_feature_importance_plot, create_diversity_plot,
    create_feature_overlap_heatmap
)

from sklearn.ensemble import RandomForestClassifier
bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@bp.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload or selection of existing file
    """
    try:
        # Check if this is a selection of existing file
        if request.is_json:
            data = request.get_json()
            if 'existing_file' in data:
                filename = secure_filename(data['existing_file'])
                filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                
                if not os.path.exists(filepath):
                    return jsonify({'success': False, 'error': 'File not found'}), 404
                
                if not allowed_file(filename):
                    return jsonify({'success': False, 'error': 'Invalid file type'}), 400
                
                # Validate dataset
                validation = validate_dataset(filepath)
                
                if not validation['valid']:
                    return jsonify({'success': False, 'error': validation['error']}), 400
                
                # Load and get dataset info
                handler = DatasetHandler(filepath)
                handler.load_data()
                handler.preprocess()
                dataset_info = handler.get_info()
                
                return jsonify({
                    'success': True,
                    'filename': filename,
                    'filepath': filepath,
                    'dataset_info': dataset_info,
                    'validation': validation
                })
        
        # Handle new file upload
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type. Only CSV files are allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Validate dataset
        validation = validate_dataset(filepath)
        
        if not validation['valid']:
            os.remove(filepath)
            return jsonify({'success': False, 'error': validation['error']}), 400
        
        # Load and get dataset info
        handler = DatasetHandler(filepath)
        handler.load_data()
        handler.preprocess()
        dataset_info = handler.get_info()
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath,
            'dataset_info': dataset_info,
            'validation': validation
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/analyze', methods=['POST'])
def analyze():
    """
    Run feature selection analysis
    """
    try:
        data = request.get_json()
        
        filepath = data.get('filepath')
        target_column = data.get('target_column')
        
        # GA Parameters
        population_size = int(data.get('population_size', 50))
        generations = int(data.get('generations', 100))
        crossover_rate = float(data.get('crossover_rate', 0.8))
        mutation_rate = float(data.get('mutation_rate', 0.1))
        elite_size = int(data.get('elite_size', 5))
        n_jobs = int(data.get('n_jobs', -1))  # -1 means use all CPU cores
        
        # Load and preprocess dataset
        handler = DatasetHandler(filepath, target_column)
        handler.load_data()
        X, y = handler.preprocess()
        feature_names = handler.get_feature_names()
        
        # Run Genetic Algorithm
        ga = GeneticAlgorithm(
            X, y,
            population_size=population_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elite_size=elite_size,
            n_jobs=n_jobs
        )
        
        ga_results = ga.run(verbose=False)
        
        # Run Statistical Methods
        stat_methods = StatisticalMethods(X, y, feature_names)
        stat_results = stat_methods.run_all_methods(verbose=False)
        
        # Compare results
        comparison = stat_methods.compare_with_ga(ga_results['selected_features'])
        
        # Create visualizations
        convergence_plot = create_convergence_plot(ga_results['fitness_history'])
        diversity_plot = create_diversity_plot(ga_results['diversity_history'])
        comparison_plot = create_comparison_plot(comparison)
        overlap_heatmap = create_feature_overlap_heatmap(comparison, feature_names)
        
        # Create feature importance plots for each method
        feature_plots = {}
        
        # GA selected features
        selected_idx = ga_results['selected_features']
        ga_importance_full = [0.0] * len(feature_names)
        if len(selected_idx) > 0:
            clf_imp = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            X_sel = X[:, selected_idx]
            clf_imp.fit(X_sel, y)
            importances = clf_imp.feature_importances_
            for idx, imp in zip(selected_idx, importances):
                ga_importance_full[idx] = float(imp)
        feature_plots['ga'] = create_feature_importance_plot(
            list(range(len(ga_importance_full))),
            feature_names,
            ga_importance_full,
            'Genetic Algorithm'
        )
        
        # Statistical methods
        for method_key, method_data in stat_results.items():
            if 'scores' in method_data:
                feature_plots[method_key] = create_feature_importance_plot(
                    list(range(len(method_data['scores']))),
                    feature_names,
                    method_data['scores'],
                    method_data['method']
                )
        
        # Prepare response
        response = {
            'success': True,
            'ga_results': {
                'selected_features': ga_results['selected_features'],
                'selected_feature_names': [feature_names[i] for i in ga_results['selected_features']],
                'n_selected': ga_results['n_selected'],
                'n_total': ga_results['n_total'],
                'best_fitness': ga_results['best_fitness'],
                'execution_time': ga_results['execution_time'],
                'parameters': ga_results['parameters']
            },
            'statistical_results': {
                method_key: {
                    'method': method_data['method'],
                    'top_features': method_data.get('top_features', [])[:10],
                    'top_feature_names': method_data.get('feature_names', [])[:10],
                    'execution_time': method_data['execution_time']
                }
                for method_key, method_data in stat_results.items()
            },
            'comparison': comparison,
            'visualizations': {
                'convergence': convergence_plot,
                'diversity': diversity_plot,
                'comparison': comparison_plot,
                'overlap_heatmap': overlap_heatmap,
                'feature_importance': feature_plots
            }
        }
        
        # Save results
        results_filename = f"results_{int(time.time())}.json"
        results_path = os.path.join(current_app.config['RESULTS_FOLDER'], results_filename)
        
        with open(results_path, 'w') as f:
            json.dump(response, f, indent=2)
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/results/<filename>')
def get_results(filename):
    """
    Retrieve saved results
    """
    try:
        results_path = os.path.join(current_app.config['RESULTS_FOLDER'], filename)
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/datasets')
def list_datasets():
    """
    List all uploaded datasets
    """
    try:
        upload_folder = current_app.config['UPLOAD_FOLDER']
        files = []
        
        for filename in os.listdir(upload_folder):
            if allowed_file(filename):
                filepath = os.path.join(upload_folder, filename)
                files.append({
                    'filename': filename,
                    'size': os.path.getsize(filepath),
                    'modified': os.path.getmtime(filepath)
                })
        
        return jsonify({'success': True, 'files': files})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/health')
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time()
    })
