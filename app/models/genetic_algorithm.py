import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import time
from multiprocessing import Pool, cpu_count
from functools import partial


def _evaluate_chromosome_wrapper(chromosome, X, y, n_features, random_state):
    """
    Wrapper function for parallel fitness evaluation
    This needs to be at module level for multiprocessing to work
    """
    selected_indices = np.where(chromosome == 1)[0]
    
    if len(selected_indices) == 0:
        return 0.0
    
    X_selected = X[:, selected_indices]
    
    try:
        # Use Random Forest for fitness evaluation
        clf = RandomForestClassifier(n_estimators=20, max_depth=10, 
                                    random_state=random_state, n_jobs=1)
        
        # Cross-validation score
        scores = cross_val_score(clf, X_selected, y, cv=2, scoring='accuracy')
        accuracy = scores.mean()
        
        # Penalty for using too many features
        feature_ratio = len(selected_indices) / n_features
        alpha = 0.05  # penalty weight
        
        fitness = accuracy - (alpha * feature_ratio)
        
        return fitness
    except:
        return 0.0


class GeneticAlgorithm:
    """
    Genetic Algorithm for Feature Selection
    
    Uses evolutionary principles to find optimal feature subsets:
    - Chromosome: Binary vector representing feature selection
    - Fitness: Model accuracy with penalty for too many features
    - Operations: Selection, Crossover, Mutation, Elitism
    """
    
    def __init__(self, X, y, population_size=50, generations=100, 
                 crossover_rate=0.8, mutation_rate=0.1, elite_size=5,
                 tournament_size=3, random_state=42, n_jobs=-1):
        """
        Initialize Genetic Algorithm
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        population_size : int
            Number of chromosomes in population
        generations : int
            Number of evolutionary iterations
        crossover_rate : float
            Probability of crossover (0-1)
        mutation_rate : float
            Probability of mutation (0-1)
        elite_size : int
            Number of best solutions to preserve
        tournament_size : int
            Number of individuals in tournament selection
        random_state : int
            Random seed for reproducibility
        n_jobs : int
            Number of CPU cores to use (-1 for all cores)
        """
        self.X = X
        self.y = y
        self.n_features = X.shape[1]
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.random_state = random_state
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        
        np.random.seed(random_state)
        
        self.population = None
        self.fitness_scores = None
        self.best_chromosome = None
        self.best_fitness = -np.inf
        self.fitness_history = []
        self.diversity_history = []
        self.selected_features = None
        self.execution_time = 0
        
    def initialize_population(self):
        """
        Create initial population with random chromosomes
        Ensures at least one feature is selected per chromosome
        """
        population = []
        for _ in range(self.population_size):
            chromosome = np.random.randint(0, 2, self.n_features)
            # Ensure at least one feature is selected
            if chromosome.sum() == 0:
                chromosome[np.random.randint(0, self.n_features)] = 1
            population.append(chromosome)
        self.population = np.array(population)
        
    def calculate_fitness(self, chromosome):
        """
        Calculate fitness score for a chromosome
        
        Fitness = Accuracy - (alpha * feature_ratio)
        where feature_ratio penalizes using too many features
        """
        selected_indices = np.where(chromosome == 1)[0]
        
        if len(selected_indices) == 0:
            return 0.0
        
        X_selected = self.X[:, selected_indices]
        
        try:
            # Use Random Forest for fitness evaluation
            # Optimized: Reduced estimators (50→20) and CV folds (3→2) for faster evaluation
            # Use n_jobs=1 here since parallelization happens at population level
            clf = RandomForestClassifier(n_estimators=20, max_depth=10, 
                                        random_state=self.random_state, n_jobs=1)
            
            # Cross-validation score
            scores = cross_val_score(clf, X_selected, self.y, cv=2, scoring='accuracy')
            accuracy = scores.mean()
            
            # Penalty for using too many features
            feature_ratio = len(selected_indices) / self.n_features
            alpha = 0.05  # penalty weight
            
            fitness = accuracy - (alpha * feature_ratio)
            
            return fitness
        except:
            return 0.0
    
    def evaluate_population(self):
        """Evaluate fitness for entire population using parallel processing"""
        if self.n_jobs > 1:
            # Parallel evaluation using all CPU cores
            with Pool(processes=self.n_jobs) as pool:
                # Create partial function with fixed X, y, n_features, and random_state
                evaluate_func = partial(_evaluate_chromosome_wrapper, 
                                      X=self.X, y=self.y, 
                                      n_features=self.n_features,
                                      random_state=self.random_state)
                self.fitness_scores = np.array(pool.map(evaluate_func, self.population))
        else:
            # Sequential evaluation (fallback)
            self.fitness_scores = np.array([self.calculate_fitness(chrom) 
                                           for chrom in self.population])
        
        # Track best solution
        max_idx = np.argmax(self.fitness_scores)
        if self.fitness_scores[max_idx] > self.best_fitness:
            self.best_fitness = self.fitness_scores[max_idx]
            self.best_chromosome = self.population[max_idx].copy()
    
    def tournament_selection(self):
        """
        Select parent using tournament selection
        Randomly pick tournament_size individuals and select the best
        """
        tournament_indices = np.random.choice(self.population_size, 
                                             self.tournament_size, 
                                             replace=False)
        tournament_fitness = self.fitness_scores[tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[winner_idx].copy()
    
    def crossover(self, parent1, parent2):
        """
        Single-point crossover
        Randomly select a crossover point and swap genes
        """
        if np.random.random() < self.crossover_rate:
            point = np.random.randint(1, self.n_features)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def mutate(self, chromosome):
        """
        Bit-flip mutation
        Randomly flip bits with probability mutation_rate
        """
        for i in range(len(chromosome)):
            if np.random.random() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i]
        
        # Ensure at least one feature is selected
        if chromosome.sum() == 0:
            chromosome[np.random.randint(0, self.n_features)] = 1
            
        return chromosome
    
    def calculate_diversity(self):
        """
        Calculate population diversity
        Measures how different individuals are from each other
        """
        diversity = 0
        for i in range(self.population_size):
            for j in range(i + 1, self.population_size):
                diversity += np.sum(self.population[i] != self.population[j])
        
        max_diversity = (self.population_size * (self.population_size - 1) / 2) * self.n_features
        return diversity / max_diversity if max_diversity > 0 else 0
    
    def evolve(self):
        """
        Create next generation using genetic operations
        """
        new_population = []
        
        # Elitism: preserve best solutions
        elite_indices = np.argsort(self.fitness_scores)[-self.elite_size:]
        for idx in elite_indices:
            new_population.append(self.population[idx].copy())
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # Crossover
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutation
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        self.population = np.array(new_population[:self.population_size])
    
    def run(self, verbose=True):
        """
        Execute the genetic algorithm
        
        Returns:
        --------
        dict: Results containing best features, fitness, and history
        """
        start_time = time.time()
        
        if verbose:
            print(f"Using {self.n_jobs} CPU cores for parallel processing")
        
        # Initialize
        self.initialize_population()
        self.evaluate_population()
        
        # Evolution loop
        for generation in range(self.generations):
            # Track statistics
            self.fitness_history.append({
                'generation': generation,
                'best_fitness': self.best_fitness,
                'avg_fitness': np.mean(self.fitness_scores),
                'worst_fitness': np.min(self.fitness_scores)
            })
            
            diversity = self.calculate_diversity()
            self.diversity_history.append(diversity)
            
            if verbose and generation % 10 == 0:
                selected_count = np.sum(self.best_chromosome)
                print(f"Generation {generation}: "
                      f"Best Fitness = {self.best_fitness:.4f}, "
                      f"Features = {selected_count}/{self.n_features}, "
                      f"Diversity = {diversity:.4f}")
            
            # Create next generation
            self.evolve()
            self.evaluate_population()
        
        self.execution_time = time.time() - start_time
        
        # Extract selected features
        self.selected_features = np.where(self.best_chromosome == 1)[0]
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Genetic Algorithm Completed!")
            print(f"{'='*60}")
            print(f"Execution Time: {self.execution_time:.2f} seconds")
            print(f"Best Fitness: {self.best_fitness:.4f}")
            print(f"Selected Features: {len(self.selected_features)}/{self.n_features}")
            print(f"Feature Indices: {self.selected_features.tolist()}")
        
        return self.get_results()
    
    def get_results(self):
        """
        Get comprehensive results from the GA run
        """
        return {
            'selected_features': self.selected_features.tolist(),
            'n_selected': len(self.selected_features),
            'n_total': self.n_features,
            'best_fitness': float(self.best_fitness),
            'best_chromosome': self.best_chromosome.tolist(),
            'fitness_history': self.fitness_history,
            'diversity_history': self.diversity_history,
            'execution_time': self.execution_time,
            'parameters': {
                'population_size': self.population_size,
                'generations': self.generations,
                'crossover_rate': self.crossover_rate,
                'mutation_rate': self.mutation_rate,
                'elite_size': self.elite_size,
                'tournament_size': self.tournament_size
            }
        }
