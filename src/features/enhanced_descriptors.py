"""
Enhanced Molecular Descriptor Generator for MDM2 Inhibitor Prediction
Implements comprehensive feature engineering with fingerprints, 3D descriptors, and binding-relevant features
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import CalcNumAromaticRings, CalcNumAliphaticRings
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.EState import Fingerprinter
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import logging
from typing import List, Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMolecularDescriptors:
    """Comprehensive molecular descriptor generator for MDM2 binding prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca_reducer = None
        self.variance_selector = None
        self.feature_selector = None
        self.feature_names = None
        
        # Initialize RDKit descriptors
        self.descriptor_names = [name[0] for name in Descriptors._descList]
        self.descriptor_calc = MoleculeDescriptors.MolecularDescriptorCalculator(self.descriptor_names)
        
        # Initialize fingerprint generators with correct parameters
        self.ecfp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
        
        # Pharmacophore features
        fdef_file = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        if os.path.exists(fdef_file):
            self.feat_factory = ChemicalFeatures.BuildFeatureFactory(fdef_file)
        else:
            self.feat_factory = None
            logger.warning("Pharmacophore feature factory not available")
    
    def compute_2d_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """Compute comprehensive 2D molecular descriptors"""
        descriptors = {}
        
        try:
            # Basic RDKit descriptors
            rdkit_values = self.descriptor_calc.CalcDescriptors(mol)
            for name, value in zip(self.descriptor_names, rdkit_values):
                if not np.isnan(value) and not np.isinf(value):
                    descriptors[f'rdkit_{name.lower()}'] = float(value)
            
            # Additional binding-relevant descriptors
            descriptors.update({
                'molecular_weight': Descriptors.MolWt(mol),
                'alogp': Descriptors.MolLogP(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'psa': Descriptors.TPSA(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'aromatic_rings': CalcNumAromaticRings(mol),
                'aliphatic_rings': CalcNumAliphaticRings(mol),
                'heavy_atoms': Descriptors.HeavyAtomCount(mol),
                'fraction_csp3': Descriptors.FractionCsp3(mol),
                'ring_count': Descriptors.RingCount(mol),
                'molar_refractivity': Descriptors.MolMR(mol),
                'balaban_j': Descriptors.BalabanJ(mol),
                'bertz_ct': Descriptors.BertzCT(mol),
                'qed': Descriptors.qed(mol),
                
                # Lipinski rule descriptors
                'lipinski_violations': sum([
                    Descriptors.MolWt(mol) > 500,
                    Descriptors.MolLogP(mol) > 5,
                    Descriptors.NumHAcceptors(mol) > 10,
                    Descriptors.NumHDonors(mol) > 5
                ]),
                
                # Additional binding-relevant features
                'formal_charge': Chem.rdmolops.GetFormalCharge(mol),
                'num_radical_electrons': Descriptors.NumRadicalElectrons(mol),
                'num_valence_electrons': Descriptors.NumValenceElectrons(mol),
                
                # Atom type counts
                'num_carbon': sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6),
                'num_nitrogen': sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7),
                'num_oxygen': sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8),
                'num_sulfur': sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 16),
                'num_halogen': sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in [9, 17, 35, 53]),
                
                # Flexibility and shape descriptors
                'kappa1': Descriptors.Kappa1(mol),
                'kappa2': Descriptors.Kappa2(mol),
                'kappa3': Descriptors.Kappa3(mol),
                'asphericity': Descriptors.Asphericity(mol),
                'eccentricity': Descriptors.Eccentricity(mol),
                'inertial_shape_factor': Descriptors.InertialShapeFactor(mol),
                'radius_of_gyration': Descriptors.RadiusOfGyration(mol),
                'spherocity_index': Descriptors.SpherocityIndex(mol)
            })
            
        except Exception as e:
            logger.warning(f"Error computing 2D descriptors: {e}")
        
        return descriptors
    
    def compute_fingerprints(self, mol: Chem.Mol) -> Dict[str, Union[float, List[int]]]:
        """Compute molecular fingerprints"""
        fingerprints = {}
        
        try:
            # Morgan/ECFP fingerprint (circular)
            ecfp = self.ecfp_gen.GetFingerprint(mol)
            fingerprints['ecfp'] = [int(x) for x in ecfp.ToBitString()]
            
            # MACCS keys (using direct RDKit function)
            maccs = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
            fingerprints['maccs'] = [int(x) for x in maccs.ToBitString()]
            
            # Estate fingerprint
            estate = Fingerprinter.FingerprintMol(mol)[0]
            fingerprints['estate'] = estate
            
            # Atom pair fingerprint
            ap = rdMolDescriptors.GetHashedAtomPairFingerprint(mol, nBits=1024)
            fingerprints['atompair'] = [int(x) for x in ap.ToBitString()]
            
            # Topological torsion fingerprint  
            tt = rdMolDescriptors.GetHashedTopologicalTorsionFingerprint(mol, nBits=1024)
            fingerprints['torsion'] = [int(x) for x in tt.ToBitString()]
            
        except Exception as e:
            logger.warning(f"Error computing fingerprints: {e}")
        
        return fingerprints
    
    def compute_3d_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """Compute 3D molecular descriptors (requires conformer generation)"""
        descriptors_3d = {}
        
        try:
            # Generate 3D conformer
            mol_copy = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_copy, randomSeed=42)
            AllChem.UFFOptimizeMolecule(mol_copy)
            
            if mol_copy.GetNumConformers() > 0:
                # 3D shape descriptors
                descriptors_3d.update({
                    'pmi1': Descriptors.PMI1(mol_copy),
                    'pmi2': Descriptors.PMI2(mol_copy),
                    'pmi3': Descriptors.PMI3(mol_copy),
                    'npr1': Descriptors.NPR1(mol_copy),
                    'npr2': Descriptors.NPR2(mol_copy),
                    'plane_of_best_fit': Descriptors.PBF(mol_copy),
                    
                    # 3D autocorrelation descriptors
                    'mor01': rdMolDescriptors.CalcMORSE(mol_copy, 0)[0] if len(rdMolDescriptors.CalcMORSE(mol_copy, 0)) > 0 else 0,
                    'mor02': rdMolDescriptors.CalcMORSE(mol_copy, 1)[0] if len(rdMolDescriptors.CalcMORSE(mol_copy, 1)) > 0 else 0,
                    
                    # RDF descriptors (Radial Distribution Function)
                    'rdf010': rdMolDescriptors.CalcRDF(mol_copy)[0] if len(rdMolDescriptors.CalcRDF(mol_copy)) > 0 else 0,
                    'rdf020': rdMolDescriptors.CalcRDF(mol_copy)[1] if len(rdMolDescriptors.CalcRDF(mol_copy)) > 1 else 0,
                    
                    # WHIM descriptors (sample)
                    'whim1': rdMolDescriptors.CalcWHIM(mol_copy)[0] if len(rdMolDescriptors.CalcWHIM(mol_copy)) > 0 else 0,
                    'whim2': rdMolDescriptors.CalcWHIM(mol_copy)[1] if len(rdMolDescriptors.CalcWHIM(mol_copy)) > 1 else 0,
                })
                
        except Exception as e:
            logger.warning(f"Error computing 3D descriptors: {e}")
            # Fill with default values if 3D computation fails
            descriptors_3d = {
                'pmi1': 0.0, 'pmi2': 0.0, 'pmi3': 0.0,
                'npr1': 0.0, 'npr2': 0.0, 'plane_of_best_fit': 0.0,
                'mor01': 0.0, 'mor02': 0.0, 'rdf010': 0.0, 'rdf020': 0.0,
                'whim1': 0.0, 'whim2': 0.0
            }
        
        return descriptors_3d
    
    def compute_pharmacophore_features(self, mol: Chem.Mol) -> Dict[str, float]:
        """Compute pharmacophore-based descriptors"""
        pharm_descriptors = {}
        
        if self.feat_factory is None:
            return {'pharm_donor': 0, 'pharm_acceptor': 0, 'pharm_hydrophobic': 0, 
                   'pharm_aromatic': 0, 'pharm_positive': 0, 'pharm_negative': 0}
        
        try:
            features = self.feat_factory.GetFeaturesForMol(mol)
            
            feature_counts = {
                'pharm_donor': 0,
                'pharm_acceptor': 0,
                'pharm_hydrophobic': 0,
                'pharm_aromatic': 0,
                'pharm_positive': 0,
                'pharm_negative': 0
            }
            
            for feat in features:
                feat_type = feat.GetType().lower()
                if 'donor' in feat_type:
                    feature_counts['pharm_donor'] += 1
                elif 'acceptor' in feat_type:
                    feature_counts['pharm_acceptor'] += 1
                elif 'hydrophobic' in feat_type:
                    feature_counts['pharm_hydrophobic'] += 1
                elif 'aromatic' in feat_type:
                    feature_counts['pharm_aromatic'] += 1
                elif 'positive' in feat_type:
                    feature_counts['pharm_positive'] += 1
                elif 'negative' in feat_type:
                    feature_counts['pharm_negative'] += 1
            
            pharm_descriptors = feature_counts
            
        except Exception as e:
            logger.warning(f"Error computing pharmacophore features: {e}")
            pharm_descriptors = {'pharm_donor': 0, 'pharm_acceptor': 0, 'pharm_hydrophobic': 0,
                               'pharm_aromatic': 0, 'pharm_positive': 0, 'pharm_negative': 0}
        
        return pharm_descriptors
    
    def process_molecules(self, smiles_list: List[str], include_3d: bool = True,
                         include_fingerprints: bool = True) -> pd.DataFrame:
        """Process molecules to generate comprehensive descriptor matrix"""
        logger.info(f"Processing {len(smiles_list)} molecules for enhanced descriptors...")
        
        all_descriptors = []
        
        for i, smiles in enumerate(smiles_list):
            if i % 100 == 0:
                logger.info(f"Processing molecule {i+1}/{len(smiles_list)}")
            
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    logger.warning(f"Invalid SMILES: {smiles}")
                    continue
                
                # Compute different types of descriptors
                mol_descriptors = {'smiles': smiles}
                
                # 2D descriptors
                descriptors_2d = self.compute_2d_descriptors(mol)
                mol_descriptors.update(descriptors_2d)
                
                # Pharmacophore features
                pharm_features = self.compute_pharmacophore_features(mol)
                mol_descriptors.update(pharm_features)
                
                # 3D descriptors (optional, computationally expensive)
                if include_3d:
                    descriptors_3d = self.compute_3d_descriptors(mol)
                    mol_descriptors.update(descriptors_3d)
                
                # Fingerprints (optional, high-dimensional)
                if include_fingerprints:
                    fingerprints = self.compute_fingerprints(mol)
                    
                    # Convert bit strings to individual features
                    for fp_type, fp_data in fingerprints.items():
                        if isinstance(fp_data, list) and fp_type in ['ecfp', 'maccs', 'atompair', 'torsion']:
                            # Convert bit string to individual binary features
                            for bit_idx, bit_val in enumerate(fp_data):
                                mol_descriptors[f'{fp_type}_bit_{bit_idx:04d}'] = int(bit_val)
                        elif fp_type == 'estate':
                            # Estate fingerprint is already a list of counts
                            for j, val in enumerate(fp_data):
                                mol_descriptors[f'estate_{j:03d}'] = float(val)
                
                all_descriptors.append(mol_descriptors)
                
            except Exception as e:
                logger.warning(f"Error processing SMILES {smiles}: {e}")
                continue
        
        if not all_descriptors:
            logger.error("No valid descriptors computed")
            return pd.DataFrame()
        
        # Convert to DataFrame
        descriptor_df = pd.DataFrame(all_descriptors)
        
        # Handle missing values
        numeric_columns = descriptor_df.select_dtypes(include=[np.number]).columns
        descriptor_df[numeric_columns] = descriptor_df[numeric_columns].fillna(0)
        
        logger.info(f"Generated {len(descriptor_df)} descriptor vectors with {len(descriptor_df.columns)-1} features")
        
        return descriptor_df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       max_features: int = 500,
                       variance_threshold: float = 0.01) -> Tuple[pd.DataFrame, List[str]]:
        """Intelligent feature selection for enhanced descriptors"""
        logger.info("Performing feature selection on enhanced descriptors...")
        
        # Remove SMILES column for feature selection
        feature_columns = [col for col in X.columns if col != 'smiles']
        X_features = X[feature_columns].copy()
        
        # 1. Remove low-variance features
        self.variance_selector = VarianceThreshold(threshold=variance_threshold)
        X_var_selected = self.variance_selector.fit_transform(X_features)
        selected_features_var = X_features.columns[self.variance_selector.get_support()]
        
        logger.info(f"After variance threshold: {len(selected_features_var)} features")
        
        # 2. Statistical feature selection
        if len(selected_features_var) > max_features:
            self.feature_selector = SelectKBest(score_func=f_classif, k=max_features)
            X_selected = self.feature_selector.fit_transform(X_var_selected, y)
            
            # Get selected feature names
            selected_indices = self.feature_selector.get_support(indices=True)
            self.selected_feature_names = [selected_features_var[i] for i in selected_indices]
        else:
            self.selected_feature_names = list(selected_features_var)
            X_selected = X_var_selected
        
        logger.info(f"Final selected features: {len(self.selected_feature_names)}")
        
        # Create final feature matrix
        X_final = pd.DataFrame(X_selected, columns=self.selected_feature_names, index=X.index)
        
        return X_final, self.selected_feature_names
    
    def transform_new_molecules(self, smiles_list: List[str], 
                              include_3d: bool = True,
                              include_fingerprints: bool = True) -> pd.DataFrame:
        """Transform new molecules using fitted feature selection"""
        if self.selected_feature_names is None:
            raise ValueError("Must fit feature selection first")
        
        # Generate descriptors
        descriptor_df = self.process_molecules(smiles_list, include_3d, include_fingerprints)
        
        # Apply same feature selection
        X_features = descriptor_df.drop('smiles', axis=1, errors='ignore')
        
        # Ensure all selected features are present
        for feature in self.selected_feature_names:
            if feature not in X_features.columns:
                X_features[feature] = 0.0
        
        # Select only the fitted features in the same order
        X_selected = X_features[self.selected_feature_names]
        
        # Apply variance threshold and statistical selection if fitted
        if self.variance_selector is not None:
            # Get the features that passed variance threshold
            var_features = X_features.columns[self.variance_selector.get_support()]
            X_var = X_features[var_features]
            
            if self.feature_selector is not None:
                X_final = self.feature_selector.transform(X_var)
                X_final = pd.DataFrame(X_final, columns=self.selected_feature_names, index=descriptor_df.index)
            else:
                X_final = X_var
        else:
            X_final = X_selected
        
        return X_final

def main():
    """Test enhanced descriptor generation"""
    descriptor_gen = EnhancedMolecularDescriptors()
    
    # Test with rigorous dataset
    logger.info("Testing enhanced descriptor generation...")
    
    # Load rigorous dataset
    train_df = pd.read_csv("data/rigorous/train_rigorous.csv")
    
    # Take a sample for testing
    sample_df = train_df.sample(n=50, random_state=42)
    
    # Process molecules with enhanced descriptors
    enhanced_descriptors = descriptor_gen.process_molecules(
        sample_df['canonical_smiles'].tolist(),
        include_3d=True,
        include_fingerprints=True
    )
    
    print(f"\nEnhanced Descriptor Generation Results:")
    print(f"Processed molecules: {len(enhanced_descriptors)}")
    print(f"Total features generated: {len(enhanced_descriptors.columns) - 1}")
    
    # Feature selection
    y = sample_df['is_inhibitor'].loc[enhanced_descriptors.index]
    X_selected, selected_features = descriptor_gen.select_features(
        enhanced_descriptors, y, max_features=100
    )
    
    print(f"Selected features: {len(selected_features)}")
    print(f"Final feature matrix shape: {X_selected.shape}")
    
    # Show feature categories
    feature_categories = {}
    for feature in selected_features:
        if 'rdkit_' in feature:
            category = 'RDKit 2D'
        elif any(fp in feature for fp in ['ecfp', 'fcfp', 'maccs', 'atompair', 'torsion']):
            category = 'Fingerprints'
        elif 'estate' in feature:
            category = 'Estate'
        elif 'pharm_' in feature:
            category = 'Pharmacophore'
        elif any(desc in feature for desc in ['pmi', 'npr', 'mor', 'rdf', 'whim']):
            category = '3D Descriptors'
        else:
            category = '2D Descriptors'
        
        if category not in feature_categories:
            feature_categories[category] = 0
        feature_categories[category] += 1
    
    print(f"\nFeature categories:")
    for category, count in feature_categories.items():
        print(f"  {category}: {count} features")
    
    # Save enhanced descriptors for rigorous dataset
    logger.info("Processing full rigorous dataset...")
    
    for split in ['train', 'val', 'test']:
        split_df = pd.read_csv(f"data/rigorous/{split}_rigorous.csv")
        
        if split == 'train':
            # Fit feature selection on training set
            enhanced_desc = descriptor_gen.process_molecules(
                split_df['canonical_smiles'].tolist(),
                include_3d=False,  # Skip 3D for speed on full dataset
                include_fingerprints=True
            )
            
            X_selected, _ = descriptor_gen.select_features(
                enhanced_desc, split_df['is_inhibitor'], max_features=200
            )
        else:
            # Transform validation/test sets
            X_selected = descriptor_gen.transform_new_molecules(
                split_df['canonical_smiles'].tolist(),
                include_3d=False,
                include_fingerprints=True
            )
        
        # Save enhanced descriptors
        enhanced_output = split_df.copy()
        enhanced_output = enhanced_output.join(X_selected, how='inner')
        enhanced_output.to_csv(f"data/rigorous/{split}_enhanced.csv", index=False)
        
        logger.info(f"Saved {split} enhanced descriptors: {X_selected.shape}")
    
    print(f"\n✅ Enhanced descriptor generation completed!")
    print(f"📊 Enhanced datasets saved to data/rigorous/")

if __name__ == "__main__":
    main()