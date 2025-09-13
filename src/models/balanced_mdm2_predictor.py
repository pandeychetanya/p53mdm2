"""
Balanced MDM2 Inhibitor Predictor
Solves the over-conservative prediction problem while maintaining high specificity
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
import logging
from typing import List, Dict, Tuple, Optional
import pickle
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BalancedMDM2Predictor:
    """Balanced predictor that solves over-conservative prediction issues"""
    
    def __init__(self, target_specificity: float = 0.85, target_sensitivity: float = 0.80):
        """
        Initialize balanced predictor
        
        Args:
            target_specificity: Target specificity (default 0.85, reduced from 0.90)
            target_sensitivity: Target sensitivity (default 0.80)
        """
        self.target_specificity = target_specificity
        self.target_sensitivity = target_sensitivity
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42, k_neighbors=3)
        
        # Initialize balanced models with different strategies
        self.base_models = {
            'balanced_rf': BalancedRandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1,
                sampling_strategy='auto'  # Automatic balancing
            ),
            'cost_sensitive_rf': RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1,
                class_weight={0: 1, 1: 3}  # Give 3x weight to inhibitors
            ),
            'balanced_lr': LogisticRegression(
                random_state=42, max_iter=2000, 
                class_weight='balanced', C=1.0, solver='liblinear'
            ),
            'cost_sensitive_svm': SVC(
                kernel='rbf', probability=True, random_state=42,
                class_weight={0: 1, 1: 2.5}, C=1.0, gamma='scale'
            )
        }
        
        self.calibrated_models = {}
        self.thresholds = {
            'conservative': 0.7,    # High specificity
            'balanced': 0.45,       # Balanced sensitivity/specificity  
            'sensitive': 0.3        # High sensitivity
        }
        self.optimal_threshold = 0.5
        
    def analyze_class_imbalance(self, y: np.ndarray) -> Dict:
        """Analyze class distribution and imbalance"""
        unique, counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        
        if len(unique) == 2:
            minority_class = min(class_distribution, key=class_distribution.get)
            majority_class = max(class_distribution, key=class_distribution.get)
            imbalance_ratio = class_distribution[majority_class] / class_distribution[minority_class]
        else:
            imbalance_ratio = 1.0
            minority_class, majority_class = None, None
        
        analysis = {
            'class_distribution': class_distribution,
            'imbalance_ratio': imbalance_ratio,
            'minority_class': minority_class,
            'majority_class': majority_class,
            'total_samples': len(y)
        }
        
        logger.info(f"Class Analysis: {class_distribution}, Imbalance Ratio: {imbalance_ratio:.1f}")
        return analysis
    
    def compute_molecular_descriptors(self, smiles_list: List[str]) -> pd.DataFrame:
        """Compute focused molecular descriptors for MDM2 binding"""
        descriptors_data = []
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    desc_dict = {name: 0.0 for name in self._get_descriptor_names()}
                else:
                    # Focus on descriptors most relevant to MDM2 binding
                    desc_dict = {
                        'molecular_weight': Descriptors.MolWt(mol),
                        'alogp': Descriptors.MolLogP(mol),
                        'hbd': Descriptors.NumHDonors(mol),
                        'hba': Descriptors.NumHAcceptors(mol),
                        'psa': Descriptors.TPSA(mol),
                        'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                        'aromatic_rings': Descriptors.NumAromaticRings(mol),
                        'heavy_atoms': Descriptors.HeavyAtomCount(mol),
                        'ring_count': Descriptors.RingCount(mol),
                        'formal_charge': Chem.rdmolops.GetFormalCharge(mol),
                        
                        # MDM2-specific binding features
                        'molar_refractivity': Descriptors.MolMR(mol),
                        'qed': Descriptors.qed(mol),
                        'bertz_ct': Descriptors.BertzCT(mol),
                        
                        # Binding pocket compatibility
                        'num_carbon': sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6),
                        'num_nitrogen': sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7),
                        'num_oxygen': sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8),
                        'num_halogen': sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in [9, 17, 35, 53]),
                        
                        # Shape and flexibility
                        'lipinski_violations': sum([
                            Descriptors.MolWt(mol) > 500,
                            Descriptors.MolLogP(mol) > 5,
                            Descriptors.NumHAcceptors(mol) > 10,
                            Descriptors.NumHDonors(mol) > 5
                        ]),
                    }
                
                desc_dict['smiles'] = smiles
                descriptors_data.append(desc_dict)
                
            except Exception as e:
                logger.warning(f"Error processing SMILES {smiles}: {e}")
                continue
        
        return pd.DataFrame(descriptors_data)
    
    def _get_descriptor_names(self) -> List[str]:
        """Get list of descriptor names"""
        return [
            'molecular_weight', 'alogp', 'hbd', 'hba', 'psa', 'rotatable_bonds',
            'aromatic_rings', 'heavy_atoms', 'ring_count', 'formal_charge',
            'molar_refractivity', 'qed', 'bertz_ct', 'num_carbon', 'num_nitrogen',
            'num_oxygen', 'num_halogen', 'lipinski_violations'
        ]
    
    def apply_smote_balancing(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE to balance training data"""
        logger.info("Applying SMOTE for class balancing...")
        
        original_counts = np.bincount(y)
        logger.info(f"Before SMOTE: {dict(enumerate(original_counts))}")
        
        try:
            # Apply SMOTE with careful parameter selection
            if len(X) < 10:  # Too few samples
                logger.warning("Too few samples for SMOTE, using original data")
                return X, y
            
            # Adjust k_neighbors based on minority class size
            minority_size = min(np.bincount(y))
            k_neighbors = min(3, minority_size - 1) if minority_size > 1 else 1
            
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
            balanced_counts = np.bincount(y_balanced)
            logger.info(f"After SMOTE: {dict(enumerate(balanced_counts))}")
            
            return X_balanced, y_balanced
            
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}, using original data")
            return X, y
    
    def train_balanced_ensemble(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
        """Train balanced ensemble addressing over-conservative predictions"""
        logger.info("Training balanced ensemble to solve over-conservative predictions...")
        
        # Analyze the problem first
        train_labels = train_df['is_inhibitor'].values
        class_analysis = self.analyze_class_imbalance(train_labels)
        
        # Compute descriptors
        train_descriptors = self.compute_molecular_descriptors(train_df['canonical_smiles'].tolist())
        val_descriptors = self.compute_molecular_descriptors(val_df['canonical_smiles'].tolist())
        
        if train_descriptors.empty or val_descriptors.empty:
            raise ValueError("Failed to compute descriptors")
        
        # Prepare features
        feature_columns = [col for col in train_descriptors.columns if col != 'smiles']
        X_train = train_descriptors[feature_columns].fillna(0)
        y_train = train_df['is_inhibitor'].values
        
        X_val = val_descriptors[feature_columns].fillna(0)
        y_val = val_df['is_inhibitor'].values
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Apply SMOTE for better class balance
        X_train_balanced, y_train_balanced = self.apply_smote_balancing(X_train_scaled, y_train)
        
        logger.info(f"Training set: {X_train_balanced.shape[0]} samples after balancing")
        
        # Train models with different balancing strategies
        training_results = {}
        
        for model_name, base_model in self.base_models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Use balanced training data
                calibrated_model = CalibratedClassifierCV(
                    base_model, method='isotonic', cv=3
                )
                calibrated_model.fit(X_train_balanced, y_train_balanced)
                
                # Evaluate on original validation set
                val_proba = calibrated_model.predict_proba(X_val_scaled)[:, 1]
                
                # Test different thresholds
                threshold_results = {}
                for threshold_name, threshold_value in self.thresholds.items():
                    val_pred = (val_proba >= threshold_value).astype(int)
                    metrics = self._calculate_metrics(y_val, val_pred, val_proba)
                    threshold_results[threshold_name] = {
                        'threshold': threshold_value,
                        'metrics': metrics
                    }
                
                self.calibrated_models[model_name] = calibrated_model
                training_results[model_name] = {
                    'threshold_results': threshold_results
                }
                
                # Log performance at different thresholds
                for thresh_name, thresh_result in threshold_results.items():
                    metrics = thresh_result['metrics']
                    logger.info(f"{model_name} ({thresh_name}): "
                              f"F1={metrics['f1_score']:.3f}, "
                              f"Spec={metrics.get('specificity', 0):.3f}, "
                              f"Sens={metrics.get('sensitivity', 0):.3f}")
                
            except Exception as e:
                logger.warning(f"Failed to train {model_name}: {e}")
                continue
        
        # Find optimal threshold balancing sensitivity and specificity
        self.optimal_threshold = self._find_balanced_threshold(X_val_scaled, y_val)
        
        # Final ensemble evaluation
        final_metrics = self._evaluate_ensemble(X_val_scaled, y_val, self.optimal_threshold)
        
        logger.info(f"Optimal threshold: {self.optimal_threshold:.3f}")
        logger.info(f"Final performance - F1: {final_metrics['f1_score']:.3f}, "
                   f"Specificity: {final_metrics.get('specificity', 0):.3f}, "
                   f"Sensitivity: {final_metrics.get('sensitivity', 0):.3f}")
        
        return {
            'class_analysis': class_analysis,
            'training_results': training_results,
            'final_metrics': final_metrics,
            'optimal_threshold': self.optimal_threshold,
            'feature_names': feature_columns
        }
    
    def _find_balanced_threshold(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Find threshold that balances sensitivity and specificity"""
        logger.info("Finding balanced threshold...")
        
        ensemble_proba = self._get_ensemble_probabilities(X_val)
        
        best_threshold = 0.5
        best_score = 0.0
        
        # Test many threshold values
        thresholds = np.arange(0.05, 0.95, 0.02)
        
        results = []
        for threshold in thresholds:
            pred = (ensemble_proba >= threshold).astype(int)
            
            cm = confusion_matrix(y_val, pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                
                # Calculate metrics
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
                
                # Balanced scoring that considers both sensitivity and specificity
                # Prefer solutions that meet minimum requirements for both
                if specificity >= (self.target_specificity - 0.05) and sensitivity >= (self.target_sensitivity - 0.1):
                    # Geometric mean of sensitivity and specificity
                    balanced_accuracy = np.sqrt(sensitivity * specificity)
                    score = balanced_accuracy + 0.1 * f1  # Small bonus for F1
                    
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
                
                results.append({
                    'threshold': threshold,
                    'specificity': specificity,
                    'sensitivity': sensitivity,
                    'f1_score': f1,
                    'balanced_accuracy': np.sqrt(sensitivity * specificity)
                })
        
        # If no threshold meets requirements, find best compromise
        if best_score == 0.0:
            logger.warning("No threshold meets target requirements, finding best compromise")
            for result in results:
                # Compromise scoring - weight both metrics
                compromise_score = (0.6 * result['specificity'] + 
                                  0.4 * result['sensitivity'] + 
                                  0.1 * result['f1_score'])
                if compromise_score > best_score:
                    best_score = compromise_score
                    best_threshold = result['threshold']
        
        logger.info(f"Selected threshold: {best_threshold:.3f} (score: {best_score:.3f})")
        return best_threshold
    
    def _get_ensemble_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Get weighted ensemble probabilities"""
        if not self.calibrated_models:
            return np.zeros(len(X))
        
        # Weight models based on their balanced performance
        model_weights = {
            'balanced_rf': 0.3,
            'cost_sensitive_rf': 0.3,
            'balanced_lr': 0.25,
            'cost_sensitive_svm': 0.15
        }
        
        ensemble_proba = np.zeros(len(X))
        total_weight = 0
        
        for model_name, model in self.calibrated_models.items():
            try:
                proba = model.predict_proba(X)[:, 1]
                weight = model_weights.get(model_name, 1.0)
                ensemble_proba += weight * proba
                total_weight += weight
            except Exception as e:
                logger.warning(f"Error with model {model_name}: {e}")
                continue
        
        return ensemble_proba / total_weight if total_weight > 0 else ensemble_proba
    
    def _evaluate_ensemble(self, X: np.ndarray, y: np.ndarray, threshold: float) -> Dict:
        """Evaluate ensemble at specific threshold"""
        ensemble_proba = self._get_ensemble_probabilities(X)
        ensemble_pred = (ensemble_proba >= threshold).astype(int)
        
        return self._calculate_metrics(y, ensemble_pred, ensemble_proba)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict:
        """Calculate comprehensive metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if len(set(y_true)) > 1:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
        
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            # Add balanced accuracy
            metrics['balanced_accuracy'] = (metrics['specificity'] + metrics['sensitivity']) / 2
        
        return metrics
    
    def predict_with_confidence(self, smiles_list: List[str], 
                              threshold_type: str = 'optimal') -> List[Dict]:
        """Make predictions with multiple threshold options"""
        
        # Compute descriptors
        descriptors = self.compute_molecular_descriptors(smiles_list)
        
        if descriptors.empty:
            return []
        
        # Prepare features
        feature_columns = [col for col in descriptors.columns if col != 'smiles']
        X = descriptors[feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Get ensemble probabilities
        ensemble_proba = self._get_ensemble_probabilities(X_scaled)
        
        # Select threshold
        if threshold_type == 'optimal':
            threshold = self.optimal_threshold
        elif threshold_type in self.thresholds:
            threshold = self.thresholds[threshold_type]
        else:
            threshold = 0.5
        
        ensemble_pred = (ensemble_proba >= threshold).astype(int)
        
        # Get individual model predictions for transparency
        individual_predictions = {}
        for model_name, model in self.calibrated_models.items():
            try:
                proba = model.predict_proba(X_scaled)[:, 1]
                individual_predictions[model_name] = proba
            except Exception as e:
                logger.warning(f"Error with model {model_name}: {e}")
        
        # Format results
        results = []
        for i, smiles in enumerate(smiles_list):
            # Calculate prediction confidence
            prob = ensemble_proba[i]
            if prob >= 0.7 or prob <= 0.3:
                confidence = 'High'
            elif prob >= 0.6 or prob <= 0.4:
                confidence = 'Medium'
            else:
                confidence = 'Low'
            
            result = {
                'smiles': smiles,
                'prediction': 'Inhibitor' if ensemble_pred[i] == 1 else 'Non-inhibitor',
                'inhibitor_probability': float(ensemble_proba[i]),
                'non_inhibitor_probability': float(1 - ensemble_proba[i]),
                'binary_prediction': int(ensemble_pred[i]),
                'confidence': confidence,
                'threshold_used': threshold,
                'threshold_type': threshold_type,
                'individual_models': {name: float(pred[i]) for name, pred in individual_predictions.items()},
                
                # Alternative predictions at different thresholds
                'alternative_predictions': {
                    'conservative': int(ensemble_proba[i] >= self.thresholds['conservative']),
                    'balanced': int(ensemble_proba[i] >= self.thresholds['balanced']),
                    'sensitive': int(ensemble_proba[i] >= self.thresholds['sensitive'])
                }
            }
            
            results.append(result)
        
        return results
    
    def save_model(self, filepath: str) -> None:
        """Save balanced model"""
        model_data = {
            'calibrated_models': self.calibrated_models,
            'scaler': self.scaler,
            'optimal_threshold': self.optimal_threshold,
            'thresholds': self.thresholds,
            'target_specificity': self.target_specificity,
            'target_sensitivity': self.target_sensitivity
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Balanced model saved to {filepath}")

def main():
    """Train and test balanced MDM2 predictor"""
    # Load rigorous dataset
    train_df = pd.read_csv("data/rigorous/train_rigorous.csv")
    val_df = pd.read_csv("data/rigorous/val_rigorous.csv")
    
    logger.info(f"Training: {len(train_df)}, Validation: {len(val_df)}")
    
    # Initialize balanced predictor with more reasonable targets
    predictor = BalancedMDM2Predictor(target_specificity=0.85, target_sensitivity=0.80)
    
    # Train balanced ensemble
    results = predictor.train_balanced_ensemble(train_df, val_df)
    
    # Save model
    predictor.save_model("models/balanced_mdm2_predictor.pkl")
    
    # Test on known compounds
    known_inhibitors = [
        "CC1=C(C=C(C=C1)C(=O)N2CCN(CC2)C(=O)C3=CC=C(C=C3)F)NC4=NC=C(C=N4)C5=CN=CC=C5",  # Milademetan
        "CC1=C(C(=NO1)C2=CC=C(C=C2)C(F)(F)F)C(=O)N3CCC(CC3)C4=NC=CN4",  # Navtemadlin
        "CC(C)NC1=NC2=C(C=CC=C2C=C1)C3=C(C=CC(=C3)Cl)C(=O)NC4CC4"  # Idasanutlin
    ]
    
    known_non_inhibitors = [
        "CCO",  # Ethanol
        "CC(=O)NC1=CC=C(C=C1)O",  # Acetaminophen
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CN(C)C(=N)N=C(N)N"  # Metformin
    ]
    
    # Test with different thresholds
    for threshold_type in ['conservative', 'balanced', 'sensitive']:
        print(f"\n{'='*60}")
        print(f"PREDICTIONS WITH {threshold_type.upper()} THRESHOLD")
        print(f"{'='*60}")
        
        inhibitor_predictions = predictor.predict_with_confidence(
            known_inhibitors, threshold_type=threshold_type
        )
        non_inhibitor_predictions = predictor.predict_with_confidence(
            known_non_inhibitors, threshold_type=threshold_type
        )
        
        print(f"\nKnown Inhibitors (should be Inhibitor):")
        for pred in inhibitor_predictions:
            print(f"  {pred['smiles'][:50]}...")
            print(f"    Prediction: {pred['prediction']}")
            print(f"    Probability: {pred['inhibitor_probability']:.3f}")
            print(f"    Confidence: {pred['confidence']}")
        
        print(f"\nKnown Non-inhibitors (should be Non-inhibitor):")
        for pred in non_inhibitor_predictions:
            print(f"  {pred['smiles']}")
            print(f"    Prediction: {pred['prediction']}")
            print(f"    Probability: {pred['inhibitor_probability']:.3f}")
            print(f"    Confidence: {pred['confidence']}")
    
    # Print final results summary
    print(f"\n{'='*80}")
    print("BALANCED MDM2 PREDICTOR RESULTS")
    print(f"{'='*80}")
    
    print(f"\nClass Balance Analysis:")
    class_analysis = results['class_analysis']
    print(f"  Original distribution: {class_analysis['class_distribution']}")
    print(f"  Imbalance ratio: {class_analysis['imbalance_ratio']:.1f}")
    
    print(f"\nFinal Performance:")
    final_metrics = results['final_metrics']
    for metric, value in final_metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    print(f"\nOptimal threshold: {results['optimal_threshold']:.3f}")
    print(f"Available thresholds: {predictor.thresholds}")
    
    print(f"\n✅ Balanced MDM2 predictor training completed!")
    print(f"💾 Model saved to models/balanced_mdm2_predictor.pkl")

if __name__ == "__main__":
    main()