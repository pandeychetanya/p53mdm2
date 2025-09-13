"""
Scaffold-Based Data Splitting for MDM2 Inhibition Prediction

This module implements Bemis-Murcko scaffold-based splitting to prevent data leakage
and ensure the model generalizes to novel scaffolds rather than memorizing known ones.

Key Features:
1. Bemis-Murcko scaffold extraction
2. Scaffold-aware train/validation/test splits
3. Challenge set generation with diverse scaffolds
4. Similarity clustering for robust evaluation
5. Data leakage prevention
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Set
import logging
import random
from itertools import combinations

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScaffoldSplitter:
    """Scaffold-based data splitting with challenge set generation"""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Storage for scaffold analysis
        self.scaffold_to_compounds = defaultdict(list)
        self.compound_to_scaffold = {}
        self.scaffold_stats = {}
        
    def extract_bemis_murcko_scaffold(self, smiles: str) -> str:
        """
        Extract Bemis-Murcko scaffold from SMILES
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Scaffold SMILES string or None if extraction fails
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
                
            # Extract Bemis-Murcko scaffold
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold is None:
                return None
                
            # Convert back to SMILES
            scaffold_smiles = Chem.MolToSmiles(scaffold)
            return scaffold_smiles
            
        except Exception as e:
            logger.warning(f"Failed to extract scaffold from {smiles}: {e}")
            return None
    
    def analyze_scaffold_diversity(self, df: pd.DataFrame, smiles_col: str = 'canonical_smiles') -> Dict:
        """
        Analyze scaffold diversity in the dataset
        
        Args:
            df: DataFrame with compounds
            smiles_col: Column name containing SMILES
            
        Returns:
            Dictionary with scaffold statistics
        """
        logger.info("Analyzing scaffold diversity...")
        
        scaffold_counts = defaultdict(int)
        scaffold_activity = defaultdict(lambda: {'active': 0, 'inactive': 0})
        
        # Extract scaffolds for all compounds
        for idx, row in df.iterrows():
            smiles = row[smiles_col]
            is_active = row.get('is_inhibitor', 0)
            
            scaffold = self.extract_bemis_murcko_scaffold(smiles)
            if scaffold:
                scaffold_counts[scaffold] += 1
                self.scaffold_to_compounds[scaffold].append(idx)
                self.compound_to_scaffold[idx] = scaffold
                
                # Track activity
                if is_active:
                    scaffold_activity[scaffold]['active'] += 1
                else:
                    scaffold_activity[scaffold]['inactive'] += 1
        
        # Calculate statistics
        total_compounds = len(df)
        total_scaffolds = len(scaffold_counts)
        singleton_scaffolds = sum(1 for count in scaffold_counts.values() if count == 1)
        
        # Most common scaffolds
        common_scaffolds = sorted(scaffold_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Scaffolds with mixed activity
        mixed_activity_scaffolds = []
        for scaffold, activity in scaffold_activity.items():
            if activity['active'] > 0 and activity['inactive'] > 0:
                mixed_activity_scaffolds.append((
                    scaffold, 
                    activity['active'], 
                    activity['inactive'],
                    scaffold_counts[scaffold]
                ))
        
        mixed_activity_scaffolds.sort(key=lambda x: x[3], reverse=True)  # Sort by total count
        
        stats = {
            'total_compounds': total_compounds,
            'total_scaffolds': total_scaffolds,
            'singleton_scaffolds': singleton_scaffolds,
            'scaffold_diversity': total_scaffolds / total_compounds,
            'common_scaffolds': common_scaffolds,
            'mixed_activity_scaffolds': mixed_activity_scaffolds[:10],
            'scaffold_counts': dict(scaffold_counts),
            'scaffold_activity': dict(scaffold_activity)
        }
        
        self.scaffold_stats = stats
        
        logger.info(f"Scaffold Analysis:")
        logger.info(f"  Total compounds: {total_compounds}")
        logger.info(f"  Total scaffolds: {total_scaffolds}")
        logger.info(f"  Singleton scaffolds: {singleton_scaffolds} ({singleton_scaffolds/total_scaffolds:.1%})")
        logger.info(f"  Scaffold diversity: {stats['scaffold_diversity']:.3f}")
        logger.info(f"  Mixed activity scaffolds: {len(mixed_activity_scaffolds)}")
        
        return stats
    
    def scaffold_split(self, df: pd.DataFrame, 
                      smiles_col: str = 'canonical_smiles',
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform scaffold-based train/validation/test split
        
        Args:
            df: DataFrame with compounds
            smiles_col: Column name containing SMILES
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation  
            test_ratio: Fraction of data for testing
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Performing scaffold-based split...")
        
        # Verify ratios sum to 1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        # Extract scaffolds if not already done
        if not self.scaffold_to_compounds:
            self.analyze_scaffold_diversity(df, smiles_col)
        
        # Get scaffold sizes and sort by size (largest first)
        scaffold_sizes = [(scaffold, len(indices)) 
                         for scaffold, indices in self.scaffold_to_compounds.items()]
        scaffold_sizes.sort(key=lambda x: x[1], reverse=True)
        
        # Assign scaffolds to splits
        train_indices = set()
        val_indices = set()
        test_indices = set()
        
        target_train = int(len(df) * train_ratio)
        target_val = int(len(df) * val_ratio)
        target_test = len(df) - target_train - target_val  # Remaining
        
        # Greedy assignment to balance split sizes
        for scaffold, size in scaffold_sizes:
            compound_indices = self.scaffold_to_compounds[scaffold]
            
            # Decide which split to assign this scaffold to
            current_train = len(train_indices)
            current_val = len(val_indices)
            current_test = len(test_indices)
            
            # Calculate how far each split is from its target
            train_deficit = max(0, target_train - current_train)
            val_deficit = max(0, target_val - current_val)
            test_deficit = max(0, target_test - current_test)
            
            # Assign to the split with the largest deficit
            if train_deficit >= val_deficit and train_deficit >= test_deficit:
                train_indices.update(compound_indices)
            elif val_deficit >= test_deficit:
                val_indices.update(compound_indices)
            else:
                test_indices.update(compound_indices)
        
        # Create split DataFrames
        train_df = df.iloc[list(train_indices)].copy()
        val_df = df.iloc[list(val_indices)].copy()
        test_df = df.iloc[list(test_indices)].copy()
        
        # Log split statistics
        logger.info(f"Scaffold split results:")
        logger.info(f"  Train: {len(train_df)} compounds ({len(train_df)/len(df):.1%})")
        logger.info(f"  Val: {len(val_df)} compounds ({len(val_df)/len(df):.1%})")
        logger.info(f"  Test: {len(test_df)} compounds ({len(test_df)/len(df):.1%})")
        
        # Log activity distribution
        for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            if 'is_inhibitor' in split_df.columns:
                active_count = split_df['is_inhibitor'].sum()
                total_count = len(split_df)
                logger.info(f"  {split_name} activity: {active_count}/{total_count} ({active_count/total_count:.1%} active)")
        
        return train_df, val_df, test_df
    
    def create_challenge_sets(self, df: pd.DataFrame,
                            smiles_col: str = 'canonical_smiles',
                            num_challenge_sets: int = 3) -> List[pd.DataFrame]:
        """
        Create challenge sets with maximally diverse scaffolds
        
        Args:
            df: DataFrame with compounds
            smiles_col: Column name containing SMILES
            num_challenge_sets: Number of challenge sets to create
            
        Returns:
            List of challenge set DataFrames
        """
        logger.info(f"Creating {num_challenge_sets} challenge sets...")
        
        # Extract scaffolds if not already done
        if not self.scaffold_to_compounds:
            self.analyze_scaffold_diversity(df, smiles_col)
        
        # Get active compounds only for challenge sets
        active_df = df[df['is_inhibitor'] == 1].copy()
        
        # Group active compounds by scaffold
        active_scaffolds = defaultdict(list)
        for idx, row in active_df.iterrows():
            if idx in self.compound_to_scaffold:
                scaffold = self.compound_to_scaffold[idx]
                active_scaffolds[scaffold].append(idx)
        
        # Create challenge sets by scaffold diversity
        challenge_sets = []
        used_scaffolds = set()
        
        for challenge_num in range(num_challenge_sets):
            challenge_indices = []
            
            # Select scaffolds not yet used
            available_scaffolds = [s for s in active_scaffolds.keys() if s not in used_scaffolds]
            
            if not available_scaffolds:
                logger.warning(f"No more unique scaffolds available for challenge set {challenge_num + 1}")
                break
            
            # Sample scaffolds for this challenge set
            num_scaffolds_per_set = max(1, len(available_scaffolds) // (num_challenge_sets - challenge_num))
            selected_scaffolds = random.sample(available_scaffolds, 
                                             min(num_scaffolds_per_set, len(available_scaffolds)))
            
            # Add compounds from selected scaffolds
            for scaffold in selected_scaffolds:
                challenge_indices.extend(active_scaffolds[scaffold])
                used_scaffolds.add(scaffold)
            
            # Create challenge set DataFrame
            challenge_df = active_df.loc[challenge_indices].copy()
            challenge_df['challenge_set'] = challenge_num + 1
            challenge_sets.append(challenge_df)
            
            logger.info(f"  Challenge Set {challenge_num + 1}: {len(challenge_df)} compounds, "
                       f"{len(selected_scaffolds)} scaffolds")
        
        return challenge_sets
    
    def calculate_scaffold_similarity_matrix(self, scaffolds: List[str]) -> np.ndarray:
        """
        Calculate scaffold-scaffold similarity matrix using fingerprints
        
        Args:
            scaffolds: List of scaffold SMILES
            
        Returns:
            Similarity matrix
        """
        logger.info("Calculating scaffold similarity matrix...")
        
        # Generate fingerprints for scaffolds
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
        fingerprints = []
        
        valid_scaffolds = []
        for scaffold in scaffolds:
            try:
                mol = Chem.MolFromSmiles(scaffold)
                if mol:
                    fp = fpgen.GetFingerprint(mol)
                    fingerprints.append(fp)
                    valid_scaffolds.append(scaffold)
            except:
                continue
        
        # Calculate similarity matrix
        n = len(fingerprints)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                    similarity_matrix[i, j] = similarity
        
        return similarity_matrix, valid_scaffolds
    
    def generate_scaffold_report(self, df: pd.DataFrame, 
                               output_path: str = None) -> Dict:
        """
        Generate comprehensive scaffold analysis report
        
        Args:
            df: DataFrame with compounds
            output_path: Path to save report plots
            
        Returns:
            Report dictionary
        """
        logger.info("Generating scaffold analysis report...")
        
        # Analyze scaffolds
        stats = self.analyze_scaffold_diversity(df)
        
        # Perform scaffold split
        train_df, val_df, test_df = self.scaffold_split(df)
        
        # Create challenge sets
        challenge_sets = self.create_challenge_sets(df)
        
        # Calculate scaffold overlap between splits
        train_scaffolds = set()
        val_scaffolds = set()
        test_scaffolds = set()
        
        for idx in train_df.index:
            if idx in self.compound_to_scaffold:
                train_scaffolds.add(self.compound_to_scaffold[idx])
        
        for idx in val_df.index:
            if idx in self.compound_to_scaffold:
                val_scaffolds.add(self.compound_to_scaffold[idx])
                
        for idx in test_df.index:
            if idx in self.compound_to_scaffold:
                test_scaffolds.add(self.compound_to_scaffold[idx])
        
        # Check for scaffold leakage
        train_val_overlap = train_scaffolds & val_scaffolds
        train_test_overlap = train_scaffolds & test_scaffolds
        val_test_overlap = val_scaffolds & test_scaffolds
        
        report = {
            'scaffold_stats': stats,
            'split_info': {
                'train_size': len(train_df),
                'val_size': len(val_df),
                'test_size': len(test_df),
                'train_scaffolds': len(train_scaffolds),
                'val_scaffolds': len(val_scaffolds),
                'test_scaffolds': len(test_scaffolds),
                'scaffold_overlap': {
                    'train_val': len(train_val_overlap),
                    'train_test': len(train_test_overlap),
                    'val_test': len(val_test_overlap)
                }
            },
            'challenge_sets': len(challenge_sets),
            'data_splits': {
                'train': train_df,
                'val': val_df,
                'test': test_df
            },
            'challenge_data': challenge_sets
        }
        
        # Log leakage check
        if any([train_val_overlap, train_test_overlap, val_test_overlap]):
            logger.warning("Scaffold leakage detected!")
            if train_val_overlap:
                logger.warning(f"  Train-Val overlap: {len(train_val_overlap)} scaffolds")
            if train_test_overlap:
                logger.warning(f"  Train-Test overlap: {len(train_test_overlap)} scaffolds")
            if val_test_overlap:
                logger.warning(f"  Val-Test overlap: {len(val_test_overlap)} scaffolds")
        else:
            logger.info("✅ No scaffold leakage detected between splits")
        
        return report

def main():
    """Main function to test scaffold splitting"""
    # Load enhanced complete dataset
    try:
        df = pd.read_csv('../../data/processed/enhanced_complete_dataset.csv')
        logger.info(f"Loaded dataset with {len(df)} compounds")
    except FileNotFoundError:
        logger.error("Enhanced complete dataset not found. Please run negative dataset expansion first.")
        return
    
    # Initialize scaffold splitter
    splitter = ScaffoldSplitter(random_seed=42)
    
    # Generate scaffold report
    report = splitter.generate_scaffold_report(df)
    
    # Save scaffold-split datasets
    train_df = report['data_splits']['train']
    val_df = report['data_splits']['val']
    test_df = report['data_splits']['test']
    
    train_df.to_csv('../../data/processed/scaffold_train.csv', index=False)
    val_df.to_csv('../../data/processed/scaffold_val.csv', index=False)
    test_df.to_csv('../../data/processed/scaffold_test.csv', index=False)
    
    # Save challenge sets
    for i, challenge_df in enumerate(report['challenge_data']):
        challenge_df.to_csv(f'../../data/processed/challenge_set_{i+1}.csv', index=False)
    
    # Save scaffold analysis
    import json
    scaffold_stats = report['scaffold_stats'].copy()
    # Remove non-serializable items
    scaffold_stats.pop('scaffold_counts', None)
    scaffold_stats.pop('scaffold_activity', None)
    
    with open('../../data/processed/scaffold_analysis.json', 'w') as f:
        json.dump({
            'scaffold_stats': scaffold_stats,
            'split_info': report['split_info']
        }, f, indent=2)
    
    logger.info("Scaffold analysis complete!")
    logger.info(f"Saved scaffold-split datasets and challenge sets")
    
    # Print final summary
    print("\n" + "="*50)
    print("SCAFFOLD-BASED SPLIT SUMMARY")
    print("="*50)
    print(f"Total compounds: {len(df)}")
    print(f"Total scaffolds: {report['scaffold_stats']['total_scaffolds']}")
    print(f"Scaffold diversity: {report['scaffold_stats']['scaffold_diversity']:.3f}")
    print(f"\nTrain: {len(train_df)} compounds ({len(train_df)/len(df):.1%})")
    print(f"Val: {len(val_df)} compounds ({len(val_df)/len(df):.1%})")
    print(f"Test: {len(test_df)} compounds ({len(test_df)/len(df):.1%})")
    print(f"\nChallenge sets: {len(report['challenge_data'])}")
    print(f"No scaffold leakage: {'✅' if not any(report['split_info']['scaffold_overlap'].values()) else '❌'}")

if __name__ == "__main__":
    main()