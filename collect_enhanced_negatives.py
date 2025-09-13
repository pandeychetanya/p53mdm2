#!/usr/bin/env python3
"""
Collect enhanced negative dataset using existing active compounds
"""

import sys
import os
sys.path.append('src')

import pandas as pd
from data.negative_dataset_expander import NegativeDatasetExpander
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load existing active compounds
    try:
        data = pd.read_csv('data/processed/balanced_mdm2_data.csv')
        actives_df = data[data['is_inhibitor'] == 1][['canonical_smiles', 'molecule_chembl_id', 'is_inhibitor']].copy()
        logger.info(f"Loaded {len(actives_df)} active compounds")
    except Exception as e:
        logger.error(f"Error loading active compounds: {e}")
        return
    
    # Initialize expander
    expander = NegativeDatasetExpander()
    
    # Create expanded negative dataset with target ratio of 4:1 (negatives:actives)
    expanded_negatives = expander.create_expanded_negative_dataset(actives_df, target_ratio=4.0)
    
    if not expanded_negatives.empty:
        # Save expanded negatives
        output_path = 'data/processed/enhanced_negative_dataset.csv'
        expanded_negatives.to_csv(output_path, index=False)
        logger.info(f"Enhanced negative dataset saved to: {output_path}")
        logger.info(f"Total negative compounds: {len(expanded_negatives)}")
        
        # Show source distribution
        source_counts = expanded_negatives['data_source'].value_counts()
        logger.info(f"Negative compound sources:\n{source_counts}")
        
        # Combine with actives for a complete dataset
        complete_dataset = pd.concat([
            actives_df.assign(data_source='mdm2_active'),
            expanded_negatives
        ], ignore_index=True)
        
        complete_output_path = 'data/processed/enhanced_complete_dataset.csv'
        complete_dataset.to_csv(complete_output_path, index=False)
        logger.info(f"Complete enhanced dataset saved to: {complete_output_path}")
        logger.info(f"Total dataset size: {len(complete_dataset)} ({len(actives_df)} actives, {len(expanded_negatives)} negatives)")
        
        # Print class distribution
        class_dist = complete_dataset['is_inhibitor'].value_counts()
        logger.info(f"Class distribution:\n{class_dist}")
        
    else:
        logger.error("Failed to create expanded negative dataset")

if __name__ == "__main__":
    main()