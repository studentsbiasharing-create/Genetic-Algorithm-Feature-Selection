import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'csv', 'txt'}
    
    # Genetic Algorithm Default Parameters
    GA_POPULATION_SIZE = 50
    GA_GENERATIONS = 100
    GA_CROSSOVER_RATE = 0.8
    GA_MUTATION_RATE = 0.1
    GA_ELITE_SIZE = 5
    GA_TOURNAMENT_SIZE = 3
    
    # Create necessary directories
    @staticmethod
    def init_app(app):
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.RESULTS_FOLDER, exist_ok=True)
