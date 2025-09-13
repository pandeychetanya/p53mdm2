"""
Data Processor for MDM2 Inhibitor Binary Classification
Prepares dataset for training with selected features and SMILES processing
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MDM2DataProcessor:
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize data processor
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.selected_features = None
        self.feature_names = None
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load MDM2 dataset
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading dataset from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} compounds")
        return df
    
    def extract_rdkit_descriptors(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        Extract additional RDKit molecular descriptors from SMILES
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            DataFrame with RDKit descriptors
        """
        logger.info("Extracting RDKit descriptors from SMILES...")
        
        descriptors_data = []
        
        for i, smiles in enumerate(smiles_list):
            if i % 100 == 0:
                logger.info(f"Processing SMILES {i}/{len(smiles_list)}")
            
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    logger.warning(f"Invalid SMILES: {smiles}")
                    descriptors_data.append({})
                    continue
                
                # Calculate RDKit descriptors
                descriptors = {
                    'mw': Descriptors.MolWt(mol),
                    'logp': Descriptors.MolLogP(mol),
                    'hbd': Descriptors.NumHDonors(mol),
                    'hba': Descriptors.NumHAcceptors(mol),
                    'tpsa': Descriptors.TPSA(mol),
                    'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                    'aromatic_rings': Descriptors.NumAromaticRings(mol),
                    'aliphatic_rings': Descriptors.NumAliphaticRings(mol),
                    'heavy_atoms': Descriptors.HeavyAtomCount(mol),
                    'fraction_csp3': Descriptors.FractionCsp3(mol),
                    'ring_count': Descriptors.RingCount(mol),
                    'molar_refractivity': Descriptors.MolMR(mol),
                    'balabanJ': Descriptors.BalabanJ(mol),
                    'bertz_ct': Descriptors.BertzCT(mol),
                    'lipinski_hba': Descriptors.NumHAcceptors(mol),
                    'lipinski_hbd': Descriptors.NumHDonors(mol),
                    'qed': Descriptors.qed(mol)
                }
                
                descriptors_data.append(descriptors)
                
            except Exception as e:
                logger.warning(f"Error processing SMILES {smiles}: {e}")
                descriptors_data.append({})
        
        return pd.DataFrame(descriptors_data)
    
    def prepare_features(self, df: pd.DataFrame, 
                        use_rdkit: bool = True,
                        selected_features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Prepare feature matrix for modeling
        
        Args:
            df: Input dataframe
            use_rdkit: Whether to extract RDKit descriptors from SMILES
            selected_features: Specific features to use (if None, use all available)
            
        Returns:
            Processed feature DataFrame
        """
        logger.info("Preparing feature matrix...")
        
        feature_df = df.copy()
        
        # Extract RDKit descriptors if requested
        if use_rdkit and 'canonical_smiles' in df.columns:
            rdkit_descriptors = self.extract_rdkit_descriptors(df['canonical_smiles'].tolist())
            feature_df = pd.concat([feature_df, rdkit_descriptors], axis=1)
        
        # Select numeric columns for features
        numeric_columns = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and ID columns
        exclude_columns = ['is_inhibitor', 'molecule_chembl_id', 'pActivity', 'standard_value']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        # Use selected features if provided
        if selected_features:
            available_features = [col for col in selected_features if col in feature_df.columns]
            logger.info(f"Using {len(available_features)} selected features out of {len(selected_features)} requested")
            feature_columns = available_features
            self.selected_features = available_features
        else:
            self.selected_features = feature_columns
        
        # Extract feature matrix
        X = feature_df[feature_columns].fillna(0)  # Fill missing values with 0
        
        self.feature_names = list(X.columns)
        logger.info(f"Prepared feature matrix: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X
    
    def prepare_dataset(self, filepath: str, 
                       selected_features: Optional[List[str]] = None,
                       use_rdkit: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """
        Prepare complete dataset for training
        
        Args:
            filepath: Path to dataset CSV
            selected_features: List of feature names to use
            use_rdkit: Whether to extract RDKit descriptors
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, train_indices, test_indices)
        """
        # Load data
        df = self.load_data(filepath)
        
        # Prepare features
        X = self.prepare_features(df, use_rdkit=use_rdkit, selected_features=selected_features)
        y = df['is_inhibitor']
        
        # Train-test split
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X, y, range(len(X)),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Dataset split: Train {len(X_train)}, Test {len(X_test)}")
        logger.info(f"Train class distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"Test class distribution: {y_test.value_counts().to_dict()}")
        
        return (X_train_scaled, X_test_scaled, y_train.values, y_test.values, 
                pd.Series(train_idx), pd.Series(test_idx))
    
    def get_feature_info(self) -> Dict:
        """Get information about selected features"""
        if self.selected_features is None:
            return {"message": "No features selected yet"}
        
        return {
            "n_features": len(self.selected_features),
            "feature_names": self.selected_features,
            "scaler_fitted": hasattr(self.scaler, 'mean_')
        }
    
    def process_new_smiles(self, smiles: str, use_rdkit: bool = False) -> np.ndarray:
        """
        Process a new SMILES string for prediction
        
        Args:
            smiles: SMILES string
            use_rdkit: Whether to use RDKit descriptors
            
        Returns:
            Processed feature vector
        """
        if self.selected_features is None:
            raise ValueError("Must prepare dataset first to set feature names")
        
        # Create temporary DataFrame
        temp_df = pd.DataFrame({'canonical_smiles': [smiles]})
        
        if use_rdkit:
            # Extract RDKit descriptors
            rdkit_desc = self.extract_rdkit_descriptors([smiles])
            temp_df = pd.concat([temp_df, rdkit_desc], axis=1)
        
        # Select features (fill missing with 0)
        feature_vector = []
        for feature in self.selected_features:
            if feature in temp_df.columns:
                value = temp_df[feature].iloc[0]
                feature_vector.append(0 if pd.isna(value) else value)
            else:
                feature_vector.append(0)
        
        # Scale using fitted scaler
        if hasattr(self.scaler, 'mean_'):
            feature_vector = self.scaler.transform([feature_vector])[0]
        
        return np.array(feature_vector)

def main():
    """Test the data processor"""
    processor = MDM2DataProcessor()
    
    # Test with evolutionary algorithm's best features
    best_features = ['alogp', 'hba', 'hbd', 'rtb', 'num_ro5_violations', 'cx_logp', 'aromatic_rings']
    
    # Prepare dataset
    X_train, X_test, y_train, y_test, train_idx, test_idx = processor.prepare_dataset(
        "../../data/raw/mdm2_test_data.csv",
        selected_features=best_features,
        use_rdkit=False
    )
    
    print(f"Feature Info: {processor.get_feature_info()}")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    # Test processing new SMILES
    test_smiles = "CCc1ccc2nc(N3CCN(C(=O)c4ccc(F)cc4)CC3)nc2c1"  # Example SMILES
    processed_features = processor.process_new_smiles(test_smiles)
    print(f"Processed SMILES features shape: {processed_features.shape}")

if __name__ == "__main__":
    main()