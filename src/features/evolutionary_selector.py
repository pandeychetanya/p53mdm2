"""
Evolutionary Algorithm for Feature Selection
Uses NSGA-II to select optimal molecular descriptors based on Pareto optimization
"""

import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import random
import logging
from typing import Tuple, List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvolutionaryFeatureSelector:
    def __init__(self, 
                 max_features: int = 30,
                 population_size: int = 100,
                 generations: int = 50,
                 crossover_prob: float = 0.7,
                 mutation_prob: float = 0.1):
        """
        Initialize evolutionary feature selector
        
        Args:
            max_features: Maximum number of features to select
            population_size: Size of population for GA
            generations: Number of generations to evolve
            crossover_prob: Probability of crossover
            mutation_prob: Probability of mutation
        """
        self.max_features = max_features
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        
        self.feature_names = None
        self.n_features = None
        self.X_scaled = None
        self.y = None
        self.scaler = StandardScaler()
        
        # Setup DEAP
        self._setup_deap()
    
    def _setup_deap(self):
        """Setup DEAP genetic algorithm components"""
        # Create fitness function (minimize error, minimize number of features)
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)
        
        self.toolbox = base.Toolbox()
        
        # Attribute generator (binary representation)
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        
        # Genetic operators
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
        self.toolbox.register("select", tools.selNSGA2)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EvolutionaryFeatureSelector':
        """
        Fit evolutionary feature selector
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            self
        """
        logger.info(f"Starting evolutionary feature selection with {len(X.columns)} features")
        
        # Store data
        self.feature_names = list(X.columns)
        self.n_features = len(self.feature_names)
        
        # Scale features
        self.X_scaled = self.scaler.fit_transform(X)
        self.y = y.values
        
        # Register individual and population with correct size
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                             self.toolbox.attr_bool, n=self.n_features)
        self.toolbox.register("population", tools.initRepeat, list, 
                             self.toolbox.individual)
        
        # Run genetic algorithm
        self._run_evolution()
        
        return self
    
    def _evaluate_individual(self, individual) -> Tuple[float, float]:
        """
        Evaluate fitness of an individual
        
        Args:
            individual: Binary vector representing feature selection
            
        Returns:
            Tuple of (error_rate, feature_ratio)
        """
        # Convert to boolean mask
        feature_mask = np.array(individual, dtype=bool)
        
        # Check if at least one feature is selected
        if not feature_mask.any():
            return (1.0, 1.0)  # Worst possible fitness
        
        # Select features
        X_selected = self.X_scaled[:, feature_mask]
        
        # Check feature limit
        n_selected = feature_mask.sum()
        if n_selected > self.max_features:
            return (1.0, 1.0)  # Penalty for exceeding feature limit
        
        try:
            # Cross-validation score
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            cv_scores = cross_val_score(clf, X_selected, self.y, cv=5, scoring='f1')
            
            # Fitness objectives
            error_rate = 1.0 - cv_scores.mean()  # Minimize error
            feature_ratio = n_selected / self.n_features  # Minimize feature count
            
            return (error_rate, feature_ratio)
            
        except Exception as e:
            logger.warning(f"Error evaluating individual: {e}")
            return (1.0, 1.0)
    
    def _run_evolution(self):
        """Run the evolutionary algorithm"""
        logger.info(f"Running evolution for {self.generations} generations")
        
        # Initialize population
        pop = self.toolbox.population(n=self.population_size)
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("min", np.min, axis=0)
        
        # Hall of fame to keep best individuals
        hof = tools.ParetoFront()
        
        # Run algorithm
        pop, logbook = algorithms.eaMuPlusLambda(
            pop, self.toolbox, 
            mu=self.population_size//2,
            lambda_=self.population_size,
            cxpb=self.crossover_prob,
            mutpb=self.mutation_prob,
            ngen=self.generations,
            stats=stats,
            halloffame=hof,
            verbose=True
        )
        
        self.final_population = pop
        self.hall_of_fame = hof
        self.logbook = logbook
        
        logger.info(f"Evolution completed. Hall of Fame size: {len(hof)}")
    
    def get_best_features(self, n_solutions: int = 5) -> List[Dict]:
        """
        Get best feature combinations from Pareto front
        
        Args:
            n_solutions: Number of solutions to return
            
        Returns:
            List of dictionaries with feature information
        """
        if not hasattr(self, 'hall_of_fame'):
            raise ValueError("Must fit selector first")
        
        solutions = []
        
        for i, individual in enumerate(self.hall_of_fame[:n_solutions]):
            feature_mask = np.array(individual, dtype=bool)
            selected_features = [self.feature_names[j] for j in range(len(feature_mask)) 
                               if feature_mask[j]]
            
            fitness = individual.fitness.values
            
            solution = {
                'rank': i + 1,
                'features': selected_features,
                'n_features': len(selected_features),
                'error_rate': fitness[0],
                'feature_ratio': fitness[1],
                'f1_score': 1.0 - fitness[0],
                'feature_mask': feature_mask
            }
            solutions.append(solution)
        
        return solutions
    
    def transform(self, X: pd.DataFrame, solution_rank: int = 1) -> np.ndarray:
        """
        Transform data using selected features
        
        Args:
            X: Input data
            solution_rank: Which solution to use (1-indexed)
            
        Returns:
            Transformed data with selected features
        """
        if not hasattr(self, 'hall_of_fame'):
            raise ValueError("Must fit selector first")
        
        individual = self.hall_of_fame[solution_rank - 1]
        feature_mask = np.array(individual, dtype=bool)
        
        X_scaled = self.scaler.transform(X)
        return X_scaled[:, feature_mask]
    
    def get_selected_feature_names(self, solution_rank: int = 1) -> List[str]:
        """
        Get names of selected features
        
        Args:
            solution_rank: Which solution to use (1-indexed)
            
        Returns:
            List of selected feature names
        """
        if not hasattr(self, 'hall_of_fame'):
            raise ValueError("Must fit selector first")
        
        individual = self.hall_of_fame[solution_rank - 1]
        feature_mask = np.array(individual, dtype=bool)
        
        return [self.feature_names[i] for i in range(len(feature_mask)) if feature_mask[i]]

def main():
    """Test the evolutionary feature selector"""
    # Load test data
    data = pd.read_csv("../../data/raw/mdm2_test_data.csv")
    
    # Prepare features (select numeric descriptor columns)
    descriptor_columns = [
        'molecular_weight', 'alogp', 'hba', 'hbd', 'psa', 'rtb', 
        'num_ro5_violations', 'qed_weighted', 'cx_most_apka', 
        'cx_most_bpka', 'cx_logp', 'cx_logd', 'aromatic_rings', 
        'heavy_atoms', 'num_alerts'
    ]
    
    # Filter available columns
    available_columns = [col for col in descriptor_columns if col in data.columns]
    X = data[available_columns].fillna(0)  # Fill missing values
    y = data['is_inhibitor']
    
    print(f"Dataset: {len(X)} samples, {len(available_columns)} features")
    print(f"Available features: {available_columns}")
    
    # Run evolutionary feature selection
    selector = EvolutionaryFeatureSelector(
        max_features=10,
        population_size=30,
        generations=20,
        crossover_prob=0.7,
        mutation_prob=0.1
    )
    
    selector.fit(X, y)
    
    # Get best solutions
    solutions = selector.get_best_features(n_solutions=3)
    
    print(f"\nBest Feature Combinations:")
    for i, sol in enumerate(solutions):
        print(f"\nSolution {sol['rank']}:")
        print(f"  Features ({sol['n_features']}): {sol['features']}")
        print(f"  F1 Score: {sol['f1_score']:.3f}")
        print(f"  Error Rate: {sol['error_rate']:.3f}")

if __name__ == "__main__":
    main()