"""
Balanced Dataset Creator for MDM2 Inhibition Prediction
Addresses class imbalance by collecting diverse negative examples
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from chembl_webresource_client.new_client import new_client
import requests
import time
import logging
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BalancedDatasetCreator:
    """Creates balanced datasets with proper negative controls"""
    
    def __init__(self):
        self.target_client = new_client.target
        self.activity_client = new_client.activity
        self.molecule_client = new_client.molecule
        
        # Known non-targets for negative controls
        self.negative_targets = {
            "CHEMBL240": "Acetylcholinesterase",  # Different mechanism
            "CHEMBL220": "Adenosine A1 receptor",  # GPCR
            "CHEMBL231": "Adenosine A2a receptor", # GPCR  
            "CHEMBL251": "Dopamine D2 receptor",   # GPCR
            "CHEMBL256": "Histamine H1 receptor"   # GPCR
        }
        
        # Known drug-like non-inhibitors for controls
        self.control_non_inhibitors = {
            "paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
            "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O", 
            "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "glucose": "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O",
            "ethanol": "CCO",
            "methanol": "CO",
            "water": "O",
            "urea": "NC(=O)N",
            "glycine": "NCC(=O)O",
            "alanine": "CC(N)C(=O)O"
        }
    
    def collect_negative_examples(self, n_negatives: int = 200) -> pd.DataFrame:
        """Collect negative examples from non-MDM2 targets"""
        logger.info("Collecting negative examples from diverse targets...")
        
        negative_data = []
        negatives_per_target = n_negatives // len(self.negative_targets)
        
        for target_id, target_name in self.negative_targets.items():
            logger.info(f"Collecting from {target_name} ({target_id})...")
            
            try:
                # Get bioactivity data for this target
                activities = self.activity_client.filter(
                    target_chembl_id=target_id,
                    type__in=['IC50', 'Ki', 'Kd', 'EC50'],
                    relation='=',
                    assay_type__in=['B', 'F'],  # Binding and functional assays
                    standard_value__isnull=False
                ).only([
                    'molecule_chembl_id', 'canonical_smiles', 'standard_value',
                    'standard_units', 'standard_type', 'pchembl_value'
                ])
                
                activities_df = pd.DataFrame(activities[:negatives_per_target * 2])
                
                if len(activities_df) > 0:
                    # Clean and process
                    activities_df = activities_df.dropna(subset=['canonical_smiles'])
                    activities_df['standard_value'] = pd.to_numeric(activities_df['standard_value'], errors='coerce')
                    activities_df = activities_df.dropna(subset=['standard_value'])
                    
                    # Add metadata
                    activities_df['target_id'] = target_id
                    activities_df['target_name'] = target_name
                    activities_df['is_inhibitor'] = 0  # All are non-MDM2 inhibitors
                    activities_df['data_source'] = 'negative_control'
                    
                    # Take random sample
                    if len(activities_df) > negatives_per_target:
                        activities_df = activities_df.sample(n=negatives_per_target, random_state=42)
                    
                    negative_data.append(activities_df)
                    logger.info(f"Collected {len(activities_df)} negatives from {target_name}")
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Failed to collect from {target_name}: {e}")
                continue
        
        if negative_data:
            return pd.concat(negative_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def add_control_molecules(self) -> pd.DataFrame:
        """Add known non-inhibitor control molecules"""
        logger.info("Adding control non-inhibitor molecules...")
        
        control_data = []
        for name, smiles in self.control_non_inhibitors.items():
            try:
                # Validate SMILES
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                
                # Calculate basic descriptors
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                
                control_data.append({
                    'molecule_chembl_id': f'CONTROL_{name.upper()}',
                    'canonical_smiles': smiles,
                    'standard_value': None,
                    'standard_units': None,
                    'standard_type': 'control',
                    'pchembl_value': None,
                    'target_id': 'CONTROL',
                    'target_name': 'Control Non-inhibitor',
                    'is_inhibitor': 0,
                    'data_source': 'control_panel',
                    'molecule_name': name,
                    'molecular_weight': mw,
                    'alogp': logp
                })
                
            except Exception as e:
                logger.warning(f"Failed to process control {name}: {e}")
                continue
        
        return pd.DataFrame(control_data)
    
    def create_balanced_dataset(self, 
                              mdm2_data_path: str, 
                              negative_ratio: float = 1.5,
                              output_path: str = None) -> pd.DataFrame:
        """
        Create balanced dataset with proper negative controls
        
        Args:
            mdm2_data_path: Path to MDM2 positive data
            negative_ratio: Ratio of negatives to positives (1.5 = 1.5x more negatives)
            output_path: Path to save balanced dataset
            
        Returns:
            Balanced DataFrame
        """
        logger.info("Creating balanced MDM2 dataset...")
        
        # Load MDM2 positive data
        mdm2_data = pd.read_csv(mdm2_data_path)
        logger.info(f"Loaded {len(mdm2_data)} MDM2 compounds")
        
        # Count positives
        n_positives = mdm2_data['is_inhibitor'].sum()
        n_current_negatives = len(mdm2_data) - n_positives
        
        logger.info(f"Current dataset: {n_positives} positives, {n_current_negatives} negatives")
        
        # Calculate needed negatives
        target_negatives = int(n_positives * negative_ratio)
        needed_negatives = max(0, target_negatives - n_current_negatives)
        
        logger.info(f"Target negatives: {target_negatives}, need to add: {needed_negatives}")
        
        # Collect additional negatives
        additional_negatives = []
        
        if needed_negatives > 10:  # Only if we need substantial negatives
            # Add control molecules first
            control_negatives = self.add_control_molecules()
            additional_negatives.append(control_negatives)
            
            # Collect from other targets
            remaining_needed = needed_negatives - len(control_negatives)
            if remaining_needed > 0:
                target_negatives_df = self.collect_negative_examples(remaining_needed)
                if not target_negatives_df.empty:
                    additional_negatives.append(target_negatives_df)
        
        # Combine all data
        all_data = [mdm2_data]
        
        if additional_negatives:
            # Align columns
            for neg_df in additional_negatives:
                # Add missing columns with defaults
                for col in mdm2_data.columns:
                    if col not in neg_df.columns:
                        neg_df[col] = None
                
                # Reorder columns to match
                neg_df = neg_df[mdm2_data.columns]
                all_data.append(neg_df)
        
        # Combine datasets
        balanced_data = pd.concat(all_data, ignore_index=True)
        
        # Final statistics
        final_positives = balanced_data['is_inhibitor'].sum()
        final_negatives = len(balanced_data) - final_positives
        
        logger.info(f"Balanced dataset created:")
        logger.info(f"  Total: {len(balanced_data)} compounds")
        logger.info(f"  Positives: {final_positives} ({final_positives/len(balanced_data):.1%})")
        logger.info(f"  Negatives: {final_negatives} ({final_negatives/len(balanced_data):.1%})")
        logger.info(f"  Ratio: 1:{final_negatives/final_positives:.1f}")
        
        # Save if requested
        if output_path:
            balanced_data.to_csv(output_path, index=False)
            logger.info(f"Saved balanced dataset to {output_path}")
        
        return balanced_data
    
    def create_specificity_test_set(self) -> pd.DataFrame:
        """Create test set with known non-inhibitors for specificity testing"""
        logger.info("Creating specificity test set...")
        
        # Extended panel of known non-inhibitors
        specificity_molecules = {
            # Simple molecules
            "water": "O",
            "ethanol": "CCO", 
            "methanol": "CO",
            "glucose": "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O",
            
            # Common drugs (non-cancer)
            "paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
            "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            
            # Amino acids
            "glycine": "NCC(=O)O",
            "alanine": "CC(N)C(=O)O",
            "leucine": "CC(C)C[C@@H](C(=O)O)N",
            
            # Neurotransmitters  
            "dopamine": "NCCc1ccc(O)c(O)c1",
            "serotonin": "NCCc1c[nH]c2ccc(O)cc12",
            "norepinephrine": "NCC(O)c1ccc(O)c(O)c1",
            
            # Vitamins
            "vitamin_c": "C([C@H]([C@H]([C@@H](C(=O)O)O)O)O)O",
            "niacin": "C1=CC(=CN=C1)C(=O)O",
            
            # Metabolites
            "pyruvate": "CC(=O)C(=O)O",
            "lactate": "CC(C(=O)O)O",
            "citric_acid": "C(C(=O)O)C(CC(=O)O)(C(=O)O)O"
        }
        
        test_data = []
        for name, smiles in specificity_molecules.items():
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                
                test_data.append({
                    'molecule_name': name,
                    'canonical_smiles': smiles,
                    'is_inhibitor': 0,  # All should be predicted as non-inhibitors
                    'expected_prediction': 'Non-inhibitor',
                    'molecule_type': self._classify_molecule_type(name),
                    'test_purpose': 'specificity_control'
                })
                
            except Exception as e:
                logger.warning(f"Failed to process {name}: {e}")
                continue
        
        test_df = pd.DataFrame(test_data)
        logger.info(f"Created specificity test set with {len(test_df)} molecules")
        
        return test_df
    
    def _classify_molecule_type(self, name: str) -> str:
        """Classify molecule type for analysis"""
        if name in ['water', 'ethanol', 'methanol']:
            return 'simple_molecule'
        elif name in ['paracetamol', 'aspirin', 'ibuprofen', 'caffeine']:
            return 'common_drug'
        elif name in ['glycine', 'alanine', 'leucine']:
            return 'amino_acid'
        elif name in ['dopamine', 'serotonin', 'norepinephrine']:
            return 'neurotransmitter'
        elif 'vitamin' in name:
            return 'vitamin'
        elif name in ['glucose']:
            return 'sugar'
        else:
            return 'metabolite'

def main():
    """Test the balanced dataset creator"""
    creator = BalancedDatasetCreator()
    
    # Create specificity test set
    specificity_test = creator.create_specificity_test_set()
    specificity_test.to_csv("data/processed/specificity_test_set.csv", index=False)
    print(f"Created specificity test set: {len(specificity_test)} molecules")
    
    # Show breakdown by type
    type_counts = specificity_test['molecule_type'].value_counts()
    print("\nSpecificity test breakdown:")
    for mol_type, count in type_counts.items():
        print(f"  {mol_type}: {count} molecules")
    
    # Test with existing MDM2 data if available
    try:
        balanced_data = creator.create_balanced_dataset(
            "data/raw/mdm2_test_data.csv",
            negative_ratio=2.0,
            output_path="data/processed/balanced_mdm2_data.csv"
        )
        print(f"\nCreated balanced dataset: {len(balanced_data)} total compounds")
        
    except FileNotFoundError:
        print("\nMDM2 data file not found - skipping balanced dataset creation")
        
    print("\nDataset balancing completed!")

if __name__ == "__main__":
    main()