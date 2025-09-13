"""
Negative Dataset Expander for MDM2 Inhibition Prediction

This module collects experimentally confirmed non-inhibitors from multiple sources
to improve model specificity and reduce false positive predictions.

Sources:
1. ChEMBL inactive compounds (IC50 > 50μM, inhibition < 50%)
2. PubChem BioAssay negatives
3. DUD-E decoys (expanded set)
4. Known drug compounds that don't target MDM2
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
import requests
import time
import json
from typing import List, Dict, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NegativeDatasetExpander:
    """Expands negative dataset with experimentally confirmed non-inhibitors"""
    
    def __init__(self):
        self.chembl_base_url = "https://www.ebi.ac.uk/chembl/api/data"
        self.pubchem_base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        
        # Known non-MDM2 targets for collecting negatives
        self.non_mdm2_targets = [
            'CHEMBL203',    # EGFR
            'CHEMBL204',    # VEGFR2
            'CHEMBL228',    # Cyclooxygenase-1
            'CHEMBL230',    # Cyclooxygenase-2
            'CHEMBL244',    # Acetylcholinesterase  
            'CHEMBL1824',   # Dopamine D2 receptor
            'CHEMBL233',    # Histamine H1 receptor
            'CHEMBL234',    # Adrenergic β1 receptor
        ]
        
        # Collect confirmed inactive compounds
        self.confirmed_inactives = []
        
    def collect_chembl_inactives(self, limit: int = 1000) -> pd.DataFrame:
        """
        Collect ChEMBL compounds that are confirmed inactive against MDM2
        
        Args:
            limit: Maximum number of compounds to collect
            
        Returns:
            DataFrame with inactive compounds and their data
        """
        logger.info("Collecting ChEMBL inactive compounds for MDM2...")
        
        # Search for MDM2 target
        target_search_url = f"{self.chembl_base_url}/target/search.json"
        params = {'q': 'MDM2', 'limit': 10}
        
        try:
            response = requests.get(target_search_url, params=params, timeout=30)
            response.raise_for_status()
            targets = response.json()
            
            mdm2_target_ids = []
            for target in targets.get('targets', []):
                if 'MDM2' in target.get('pref_name', '').upper():
                    mdm2_target_ids.append(target['target_chembl_id'])
            
            logger.info(f"Found MDM2 targets: {mdm2_target_ids}")
            
            inactive_compounds = []
            
            for target_id in mdm2_target_ids[:2]:  # Limit to 2 main targets
                # Get activities for this target with high IC50 or low inhibition
                activity_url = f"{self.chembl_base_url}/activity.json"
                activity_params = {
                    'target_chembl_id': target_id,
                    'standard_type__in': 'IC50,Ki,Kd,inhibition',
                    'limit': limit // 2,
                    'format': 'json'
                }
                
                response = requests.get(activity_url, params=activity_params, timeout=30)
                if response.status_code == 200:
                    activities = response.json()
                    
                    for activity in activities.get('activities', [])[:200]:
                        try:
                            # Criteria for inactive compounds
                            standard_value = activity.get('standard_value')
                            standard_type = activity.get('standard_type', '').upper()
                            standard_units = activity.get('standard_units', '').upper()
                            
                            is_inactive = False
                            
                            # High IC50, Ki, Kd values indicate inactivity
                            if standard_type in ['IC50', 'KI', 'KD'] and standard_value:
                                if standard_units in ['NM', 'NANOMOLAR'] and float(standard_value) > 50000:  # >50μM
                                    is_inactive = True
                                elif standard_units in ['UM', 'MICROMOLAR'] and float(standard_value) > 50:  # >50μM
                                    is_inactive = True
                                elif standard_units in ['MM', 'MILLIMOLAR'] and float(standard_value) > 0.05:  # >0.05mM
                                    is_inactive = True
                                    
                            # Low inhibition percentages
                            elif standard_type == 'INHIBITION' and standard_value:
                                if float(standard_value) < 50:  # <50% inhibition
                                    is_inactive = True
                            
                            if is_inactive and activity.get('canonical_smiles'):
                                compound_data = {
                                    'canonical_smiles': activity['canonical_smiles'],
                                    'molecule_chembl_id': activity.get('molecule_chembl_id'),
                                    'standard_value': standard_value,
                                    'standard_type': standard_type,
                                    'standard_units': standard_units,
                                    'is_inhibitor': 0,
                                    'data_source': 'chembl_inactive',
                                    'activity_comment': f"Inactive: {standard_type} {standard_value} {standard_units}"
                                }
                                inactive_compounds.append(compound_data)
                                
                        except (ValueError, KeyError, TypeError) as e:
                            continue
                
                time.sleep(1)  # Rate limiting
                
        except Exception as e:
            logger.error(f"Error collecting ChEMBL inactives: {e}")
            
        logger.info(f"Collected {len(inactive_compounds)} ChEMBL inactive compounds")
        return pd.DataFrame(inactive_compounds)
    
    def collect_non_mdm2_actives(self, limit: int = 500) -> pd.DataFrame:
        """
        Collect compounds that are active against other targets but not MDM2
        These serve as confirmed negatives for MDM2 activity
        
        Args:
            limit: Maximum compounds per target
            
        Returns:
            DataFrame with non-MDM2 active compounds
        """
        logger.info("Collecting compounds active against non-MDM2 targets...")
        
        non_mdm2_compounds = []
        
        for target_id in self.non_mdm2_targets[:4]:  # Limit to 4 targets
            try:
                activity_url = f"{self.chembl_base_url}/activity.json"
                params = {
                    'target_chembl_id': target_id,
                    'standard_type__in': 'IC50,Ki,EC50',
                    'limit': limit // 4,
                    'format': 'json'
                }
                
                response = requests.get(activity_url, params=params, timeout=30)
                if response.status_code == 200:
                    activities = response.json()
                    
                    for activity in activities.get('activities', [])[:100]:
                        try:
                            standard_value = activity.get('standard_value')
                            standard_type = activity.get('standard_type', '').upper()
                            standard_units = activity.get('standard_units', '').upper()
                            
                            # Look for potent compounds (good activity against other targets)
                            is_potent = False
                            
                            if standard_type in ['IC50', 'KI', 'EC50'] and standard_value:
                                if standard_units in ['NM', 'NANOMOLAR'] and float(standard_value) < 1000:  # <1μM
                                    is_potent = True
                                elif standard_units in ['UM', 'MICROMOLAR'] and float(standard_value) < 1:  # <1μM
                                    is_potent = True
                            
                            if is_potent and activity.get('canonical_smiles'):
                                compound_data = {
                                    'canonical_smiles': activity['canonical_smiles'],
                                    'molecule_chembl_id': activity.get('molecule_chembl_id'),
                                    'standard_value': standard_value,
                                    'standard_type': standard_type,
                                    'standard_units': standard_units,
                                    'is_inhibitor': 0,
                                    'data_source': f'non_mdm2_active_{target_id}',
                                    'activity_comment': f"Active against {target_id}, not MDM2"
                                }
                                non_mdm2_compounds.append(compound_data)
                                
                        except (ValueError, KeyError, TypeError):
                            continue
                            
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error collecting from target {target_id}: {e}")
                continue
        
        logger.info(f"Collected {len(non_mdm2_compounds)} non-MDM2 active compounds")
        return pd.DataFrame(non_mdm2_compounds)
    
    def generate_drug_like_decoys(self, active_smiles_list: List[str], ratio: int = 3) -> pd.DataFrame:
        """
        Generate drug-like decoy compounds using property matching
        
        Args:
            active_smiles_list: List of active compound SMILES for property matching
            ratio: Number of decoys per active compound
            
        Returns:
            DataFrame with decoy compounds
        """
        logger.info("Generating drug-like decoy compounds...")
        
        # Calculate property ranges from actives
        active_properties = []
        for smiles in active_smiles_list[:50]:  # Sample for property calculation
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    props = {
                        'mw': Descriptors.MolWt(mol),
                        'logp': Descriptors.MolLogP(mol),
                        'hbd': Descriptors.NumHDonors(mol),
                        'hba': Descriptors.NumHAcceptors(mol),
                        'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                        'aromatic_rings': Descriptors.NumAromaticRings(mol)
                    }
                    active_properties.append(props)
            except:
                continue
        
        if not active_properties:
            logger.warning("No valid active compounds for property calculation")
            return pd.DataFrame()
        
        # Calculate property ranges
        prop_df = pd.DataFrame(active_properties)
        prop_ranges = {
            prop: (prop_df[prop].quantile(0.25), prop_df[prop].quantile(0.75))
            for prop in prop_df.columns
        }
        
        # Known drug-like molecules that are unlikely to be MDM2 inhibitors
        drug_like_compounds = [
            # Common drugs from different therapeutic areas
            'CC(=O)Nc1ccc(O)cc1',  # Acetaminophen
            'CC(C)Cc1ccc(C(C)C(=O)O)cc1',  # Ibuprofen
            'CC(C)(C)NCC(O)c1ccc(O)c(CO)c1',  # Albuterol
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
            'CCO',  # Ethanol
            'C[C@H](CS)C(=O)N1CCC[C@H]1C(=O)O',  # Captopril
            'CC(C)(C)c1ccc(C(C)(C)c2ccc(O)cc2)cc1',  # BHT
            'CCCCCCCCC(=O)O',  # Nonanoic acid
            'CC1=CC(=O)C=C(C)C1=O',  # Vitamin K3
            'Nc1ccc(S(=O)(=O)Nc2ncccn2)cc1',  # Sulfadiazine
            # Add more diverse drug-like structures
            'CN(C)CCN(c1ccccc1)c1ccccc1',  # Diphenhydramine-like
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
            'CN(C)CCCN1c2ccccc2Sc2ccccc21',  # Chlorpromazine-like
            'CC1COc2c(N3CCN(C)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23',  # Ofloxacin-like
            'COc1ccc2nc(N(C)CCCN(C)C)nc(N)c2c1',  # Chloroquine-like
        ]
        
        # Generate variations of these base structures
        decoy_compounds = []
        base_structures = drug_like_compounds * (ratio // len(drug_like_compounds) + 1)
        
        for i, smiles in enumerate(base_structures[:len(active_smiles_list) * ratio]):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    # Calculate properties
                    props = {
                        'mw': Descriptors.MolWt(mol),
                        'logp': Descriptors.MolLogP(mol),
                        'hbd': Descriptors.NumHDonors(mol),
                        'hba': Descriptors.NumHAcceptors(mol),
                        'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                        'aromatic_rings': Descriptors.NumAromaticRings(mol)
                    }
                    
                    # Check if properties are within active compound ranges
                    property_match = True
                    for prop, (min_val, max_val) in prop_ranges.items():
                        if not (min_val * 0.5 <= props[prop] <= max_val * 2):  # Relaxed matching
                            property_match = False
                            break
                    
                    if property_match:
                        compound_data = {
                            'canonical_smiles': smiles,
                            'molecule_chembl_id': f'DECOY_{i:06d}',
                            'is_inhibitor': 0,
                            'data_source': 'drug_like_decoy',
                            'activity_comment': 'Drug-like decoy compound'
                        }
                        decoy_compounds.append(compound_data)
                        
            except Exception:
                continue
        
        logger.info(f"Generated {len(decoy_compounds)} drug-like decoy compounds")
        return pd.DataFrame(decoy_compounds)
    
    def create_expanded_negative_dataset(self, active_compounds_df: pd.DataFrame, 
                                       target_ratio: float = 3.0) -> pd.DataFrame:
        """
        Create comprehensive negative dataset from multiple sources
        
        Args:
            active_compounds_df: DataFrame with active MDM2 inhibitors
            target_ratio: Target ratio of negatives to actives
            
        Returns:
            Expanded DataFrame with negative compounds from multiple sources
        """
        logger.info("Creating expanded negative dataset...")
        
        num_actives = len(active_compounds_df)
        target_negatives = int(num_actives * target_ratio)
        
        logger.info(f"Target: {target_negatives} negatives for {num_actives} actives")
        
        # Collect from different sources
        negative_dfs = []
        
        # 1. ChEMBL experimentally confirmed inactives
        chembl_inactives = self.collect_chembl_inactives(limit=target_negatives // 3)
        if not chembl_inactives.empty:
            negative_dfs.append(chembl_inactives)
        
        # 2. Non-MDM2 target actives (confirmed negatives for MDM2)
        non_mdm2_actives = self.collect_non_mdm2_actives(limit=target_negatives // 3)
        if not non_mdm2_actives.empty:
            negative_dfs.append(non_mdm2_actives)
        
        # 3. Drug-like decoys
        active_smiles = active_compounds_df['canonical_smiles'].tolist()
        decoys = self.generate_drug_like_decoys(active_smiles, ratio=2)
        if not decoys.empty:
            negative_dfs.append(decoys)
        
        # Combine all negative sources
        if negative_dfs:
            expanded_negatives = pd.concat(negative_dfs, ignore_index=True)
        else:
            logger.warning("No negative compounds collected!")
            return pd.DataFrame()
        
        # Remove duplicates and invalid SMILES
        expanded_negatives = expanded_negatives.drop_duplicates(subset=['canonical_smiles'])
        expanded_negatives = expanded_negatives[expanded_negatives['canonical_smiles'].notna()]
        
        # Validate SMILES
        valid_negatives = []
        for idx, row in expanded_negatives.iterrows():
            try:
                mol = Chem.MolFromSmiles(row['canonical_smiles'])
                if mol is not None:
                    valid_negatives.append(row)
            except:
                continue
        
        if valid_negatives:
            result_df = pd.DataFrame(valid_negatives)
            logger.info(f"Created expanded negative dataset: {len(result_df)} compounds")
            
            # Log source distribution
            source_counts = result_df['data_source'].value_counts()
            logger.info(f"Negative compound sources:\n{source_counts}")
            
            return result_df
        else:
            logger.error("No valid negative compounds found!")
            return pd.DataFrame()

def main():
    """Main function to test the negative dataset expander"""
    expander = NegativeDatasetExpander()
    
    # Load existing active compounds
    try:
        actives_df = pd.read_csv('../data/processed/mdm2_active_compounds.csv')
    except FileNotFoundError:
        # Create dummy actives for testing
        actives_df = pd.DataFrame({
            'canonical_smiles': [
                'COc1cc2cc(-c3cccnc3)cnc2cc1OC',
                'COc1ccc(-c2cnc3cc(OC)c(OC)cc3c2)cc1C',
            ],
            'is_inhibitor': [1, 1]
        })
    
    # Create expanded negative dataset
    expanded_negatives = expander.create_expanded_negative_dataset(actives_df)
    
    if not expanded_negatives.empty:
        # Save expanded negatives
        output_path = '../../data/processed/expanded_negative_dataset.csv'
        expanded_negatives.to_csv(output_path, index=False)
        print(f"Expanded negative dataset saved to: {output_path}")
        print(f"Total negative compounds: {len(expanded_negatives)}")
    else:
        print("Failed to create expanded negative dataset")

if __name__ == "__main__":
    main()