#!/usr/bin/env python3
"""
Run scaffold-based splitting on enhanced dataset
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_scaffold(smiles):
    """Extract Bemis-Murcko scaffold"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None:
            return None
        return Chem.MolToSmiles(scaffold)
    except:
        return None

def scaffold_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Perform scaffold-based splitting"""
    logger.info("Performing scaffold-based split...")
    
    # Extract scaffolds
    scaffold_to_indices = defaultdict(list)
    
    for idx, row in df.iterrows():
        smiles = row['canonical_smiles']
        scaffold = extract_scaffold(smiles)
        if scaffold:
            scaffold_to_indices[scaffold].append(idx)
    
    # Sort scaffolds by size (largest first)
    scaffold_sizes = [(scaffold, len(indices)) for scaffold, indices in scaffold_to_indices.items()]
    scaffold_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Assign scaffolds to splits
    train_indices = []
    val_indices = []
    test_indices = []
    
    target_train = int(len(df) * train_ratio)
    target_val = int(len(df) * val_ratio)
    
    for scaffold, size in scaffold_sizes:
        indices = scaffold_to_indices[scaffold]
        
        # Decide which split to assign this scaffold
        current_train = len(train_indices)
        current_val = len(val_indices)
        current_test = len(test_indices)
        
        train_deficit = max(0, target_train - current_train)
        val_deficit = max(0, target_val - current_val)
        test_deficit = max(0, len(df) - target_train - target_val - current_test)
        
        # Assign to split with largest deficit
        if train_deficit >= val_deficit and train_deficit >= test_deficit:
            train_indices.extend(indices)
        elif val_deficit >= test_deficit:
            val_indices.extend(indices)
        else:
            test_indices.extend(indices)
    
    # Create split DataFrames
    train_df = df.iloc[train_indices].copy()
    val_df = df.iloc[val_indices].copy()
    test_df = df.iloc[test_indices].copy()
    
    logger.info(f"Split results:")
    logger.info(f"  Train: {len(train_df)} compounds ({len(train_df)/len(df):.1%})")
    logger.info(f"  Val: {len(val_df)} compounds ({len(val_df)/len(df):.1%})")
    logger.info(f"  Test: {len(test_df)} compounds ({len(test_df)/len(df):.1%})")
    
    # Check activity distribution
    for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        if 'is_inhibitor' in split_df.columns:
            active_count = split_df['is_inhibitor'].sum()
            total_count = len(split_df)
            logger.info(f"  {name} activity: {active_count}/{total_count} ({active_count/total_count:.1%} active)")
    
    return train_df, val_df, test_df

def create_challenge_sets(df, num_sets=3):
    """Create challenge sets with diverse scaffolds"""
    logger.info(f"Creating {num_sets} challenge sets...")
    
    # Get only active compounds
    active_df = df[df['is_inhibitor'] == 1].copy()
    
    # Group by scaffold
    scaffold_to_indices = defaultdict(list)
    for idx, row in active_df.iterrows():
        scaffold = extract_scaffold(row['canonical_smiles'])
        if scaffold:
            scaffold_to_indices[scaffold].append(idx)
    
    # Create challenge sets
    challenge_sets = []
    scaffolds = list(scaffold_to_indices.keys())
    
    # Divide scaffolds among challenge sets
    scaffolds_per_set = len(scaffolds) // num_sets
    
    for i in range(num_sets):
        start_idx = i * scaffolds_per_set
        end_idx = start_idx + scaffolds_per_set if i < num_sets - 1 else len(scaffolds)
        
        challenge_indices = []
        selected_scaffolds = scaffolds[start_idx:end_idx]
        
        for scaffold in selected_scaffolds:
            challenge_indices.extend(scaffold_to_indices[scaffold])
        
        challenge_df = active_df.loc[challenge_indices].copy()
        challenge_df['challenge_set'] = i + 1
        challenge_sets.append(challenge_df)
        
        logger.info(f"  Challenge Set {i+1}: {len(challenge_df)} compounds, {len(selected_scaffolds)} scaffolds")
    
    return challenge_sets

def main():
    # Load enhanced dataset
    try:
        df = pd.read_csv('data/processed/enhanced_complete_dataset.csv')
        logger.info(f"Loaded dataset with {len(df)} compounds")
    except FileNotFoundError:
        logger.error("Enhanced complete dataset not found")
        return
    
    # Analyze scaffold diversity
    scaffolds = []
    for idx, row in df.iterrows():
        scaffold = extract_scaffold(row['canonical_smiles'])
        if scaffold:
            scaffolds.append(scaffold)
    
    total_scaffolds = len(set(scaffolds))
    scaffold_diversity = total_scaffolds / len(df)
    
    logger.info(f"Scaffold analysis:")
    logger.info(f"  Total scaffolds: {total_scaffolds}")
    logger.info(f"  Scaffold diversity: {scaffold_diversity:.3f}")
    
    # Perform scaffold split
    train_df, val_df, test_df = scaffold_split(df)
    
    # Save splits
    train_df.to_csv('data/processed/scaffold_train.csv', index=False)
    val_df.to_csv('data/processed/scaffold_val.csv', index=False)
    test_df.to_csv('data/processed/scaffold_test.csv', index=False)
    
    # Create challenge sets
    challenge_sets = create_challenge_sets(df)
    
    # Save challenge sets
    for i, challenge_df in enumerate(challenge_sets):
        challenge_df.to_csv(f'data/processed/challenge_set_{i+1}.csv', index=False)
    
    # Save analysis
    analysis = {
        'total_compounds': len(df),
        'total_scaffolds': total_scaffolds,
        'scaffold_diversity': scaffold_diversity,
        'train_size': len(train_df),
        'val_size': len(val_df), 
        'test_size': len(test_df),
        'challenge_sets': len(challenge_sets)
    }
    
    with open('data/processed/scaffold_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print("\n" + "="*50)
    print("SCAFFOLD-BASED SPLIT SUMMARY")
    print("="*50)
    print(f"Total compounds: {len(df)}")
    print(f"Total scaffolds: {total_scaffolds}")
    print(f"Scaffold diversity: {scaffold_diversity:.3f}")
    print(f"\nTrain: {len(train_df)} compounds ({len(train_df)/len(df):.1%})")
    print(f"Val: {len(val_df)} compounds ({len(val_df)/len(df):.1%})")
    print(f"Test: {len(test_df)} compounds ({len(test_df)/len(df):.1%})")
    print(f"\nChallenge sets: {len(challenge_sets)}")
    print("✅ Scaffold-based splitting complete!")

if __name__ == "__main__":
    main()