"""
ChEMBL Data Collector for MDM2 Inhibitors
Collects bioactivity data for MDM2 target from ChEMBL database
"""

import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
import time
import logging
from typing import Tuple, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChEMBLDataCollector:
    def __init__(self):
        self.target_client = new_client.target
        self.activity_client = new_client.activity
        self.molecule_client = new_client.molecule
        
        # MDM2 target ID in ChEMBL
        self.mdm2_target_id = "CHEMBL2095189"
        
    def get_mdm2_bioactivity_data(self, max_compounds: int = 1000) -> pd.DataFrame:
        """
        Fetch bioactivity data for MDM2 inhibitors from ChEMBL
        
        Args:
            max_compounds: Maximum number of compounds to retrieve
            
        Returns:
            DataFrame with compound data and bioactivity measurements
        """
        logger.info(f"Fetching bioactivity data for MDM2 target: {self.mdm2_target_id}")
        
        # Get bioactivity data for MDM2
        activities = self.activity_client.filter(
            target_chembl_id=self.mdm2_target_id,
            type__in=['IC50', 'Ki', 'Kd'],
            relation='=',
            assay_type='B'  # Binding assays
        ).only([
            'molecule_chembl_id', 'canonical_smiles', 'standard_value',
            'standard_units', 'standard_type', 'pchembl_value',
            'activity_comment', 'assay_description', 'confidence_score'
        ])
        
        activities_df = pd.DataFrame(activities[:max_compounds])
        
        if activities_df.empty:
            logger.warning("No bioactivity data found for MDM2")
            return pd.DataFrame()
        
        logger.info(f"Retrieved {len(activities_df)} bioactivity records")
        
        # Clean and filter data
        activities_df = self._clean_bioactivity_data(activities_df)
        
        # Add molecular descriptors
        activities_df = self._add_molecular_descriptors(activities_df)
        
        return activities_df
    
    def _clean_bioactivity_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and filter bioactivity data"""
        logger.info("Cleaning bioactivity data...")
        
        # Remove entries without SMILES or standard_value
        df = df.dropna(subset=['canonical_smiles', 'standard_value'])
        
        # Convert standard_value to numeric
        df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
        df = df.dropna(subset=['standard_value'])
        
        # Filter for reasonable activity ranges (1nM to 100μM)
        df = df[df['standard_value'].between(1, 100000)]
        
        # Convert to pIC50/pKi if not already available
        df['pActivity'] = df.apply(self._calculate_pactivity, axis=1)
        
        # Convert pActivity to numeric, handling any non-numeric values
        df['pActivity'] = pd.to_numeric(df['pActivity'], errors='coerce')
        df = df.dropna(subset=['pActivity'])
        
        # Create binary labels (inhibitor vs non-inhibitor)
        # Threshold: pIC50/pKi >= 6 (IC50/Ki <= 1μM) = active inhibitor
        df['is_inhibitor'] = (df['pActivity'] >= 6).astype(int)
        
        logger.info(f"After cleaning: {len(df)} compounds")
        logger.info(f"Inhibitors: {df['is_inhibitor'].sum()}, Non-inhibitors: {len(df) - df['is_inhibitor'].sum()}")
        
        return df
    
    def _calculate_pactivity(self, row) -> Optional[float]:
        """Calculate pIC50/pKi from standard_value"""
        if pd.isna(row['standard_value']):
            return None
        
        # Use pchembl_value if available, otherwise calculate
        if pd.notna(row['pchembl_value']):
            return row['pchembl_value']
        
        # Convert to pIC50/pKi: -log10(IC50 in M)
        if row['standard_units'] == 'nM':
            return -np.log10(row['standard_value'] * 1e-9)
        elif row['standard_units'] == 'uM':
            return -np.log10(row['standard_value'] * 1e-6)
        elif row['standard_units'] == 'mM':
            return -np.log10(row['standard_value'] * 1e-3)
        else:
            return None
    
    def _add_molecular_descriptors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add molecular descriptors from ChEMBL"""
        logger.info("Fetching molecular descriptors...")
        
        descriptors_list = []
        total_compounds = len(df)
        
        for idx, (_, row) in enumerate(df.iterrows()):
            molecule_id = row['molecule_chembl_id']
            
            if idx % 10 == 0:
                logger.info(f"Processing descriptors: {idx}/{total_compounds}")
            
            try:
                # Get molecular descriptors from ChEMBL
                molecule_data = self.molecule_client.get(molecule_id)
                
                if molecule_data and 'molecule_properties' in molecule_data:
                    props = molecule_data['molecule_properties']
                    descriptors = {
                        'molecule_chembl_id': molecule_id,
                        'molecular_weight': props.get('full_mwt'),
                        'alogp': props.get('alogp'),
                        'hba': props.get('hba'),
                        'hbd': props.get('hbd'),
                        'psa': props.get('psa'),
                        'rtb': props.get('rtb'),
                        'ro3_pass': props.get('ro3_pass'),
                        'num_ro5_violations': props.get('num_ro5_violations'),
                        'qed_weighted': props.get('qed_weighted'),
                        'cx_most_apka': props.get('cx_most_apka'),
                        'cx_most_bpka': props.get('cx_most_bpka'),
                        'cx_logp': props.get('cx_logp'),
                        'cx_logd': props.get('cx_logd'),
                        'aromatic_rings': props.get('aromatic_rings'),
                        'heavy_atoms': props.get('heavy_atoms'),
                        'num_alerts': props.get('num_alerts')
                    }
                    descriptors_list.append(descriptors)
                
                # Rate limiting
                time.sleep(0.05)
                
            except Exception as e:
                logger.warning(f"Failed to get descriptors for {molecule_id}: {e}")
                continue
        
        if descriptors_list:
            descriptors_df = pd.DataFrame(descriptors_list)
            df = df.merge(descriptors_df, on='molecule_chembl_id', how='left')
        
        logger.info(f"Added descriptors for {len(descriptors_list)} compounds")
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filepath: str):
        """Save dataset to CSV file"""
        df.to_csv(filepath, index=False)
        logger.info(f"Dataset saved to {filepath}")

if __name__ == "__main__":
    collector = ChEMBLDataCollector()
    
    # Collect MDM2 bioactivity data
    mdm2_data = collector.get_mdm2_bioactivity_data(max_compounds=800)
    
    if not mdm2_data.empty:
        # Save to file
        collector.save_dataset(mdm2_data, "../../data/raw/mdm2_chembl_data.csv")
        
        print(f"Dataset shape: {mdm2_data.shape}")
        print(f"Inhibitors: {mdm2_data['is_inhibitor'].sum()}")
        print(f"Non-inhibitors: {len(mdm2_data) - mdm2_data['is_inhibitor'].sum()}")
    else:
        print("No data collected")