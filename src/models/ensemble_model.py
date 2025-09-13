"""
Research-Grade Ensemble Model for MDM2 Inhibitor Prediction
Implements ensemble learning with calibrated probabilities and external validation
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
from rdkit import Chem
from rdkit.Chem import Descriptors
import logging
from typing import List, Dict, Tuple, Optional
import pickle
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchGradeEnsemble:
    """Research-grade ensemble model with rigorous validation"""
    
    def __init__(self, specificity_target: float = 0.90):
        """
        Initialize ensemble model
        
        Args:
            specificity_target: Target specificity for optimization
        """
        self.specificity_target = specificity_target
        self.scaler = StandardScaler()
        
        # Initialize diverse base models
        self.base_models = {
            'random_forest': RandomForestClassifier(
                n_estimators=300, max_depth=12, min_samples_split=10,
                min_samples_leaf=5, random_state=42, n_jobs=-1,
                class_weight='balanced'
            ),
            'logistic_regression': LogisticRegression(
                random_state=42, max_iter=2000, class_weight='balanced',
                C=0.1, solver='liblinear'
            ),
            'svm': SVC(
                kernel='rbf', probability=True, random_state=42, 
                class_weight='balanced', C=1.0, gamma='scale'
            ),
        }
        
        # Add XGBoost if available
        if HAS_XGBOOST:
            self.base_models['xgboost'] = xgb.XGBClassifier(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                random_state=42, eval_metric='logloss',
                scale_pos_weight=2.0  # Give more weight to positives
            )
        
        self.calibrated_models = {}
        self.optimal_threshold = 0.5
        
    def compute_molecular_descriptors(self, smiles_list: List[str]) -> pd.DataFrame:
        """Compute molecular descriptors optimized for MDM2 binding"""
        descriptors_data = []
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    # Fill with zeros for invalid SMILES
                    desc_dict = {
                        'molecular_weight': 0.0, 'alogp': 0.0, 'hbd': 0.0, 'hba': 0.0,
                        'psa': 0.0, 'rotatable_bonds': 0.0, 'aromatic_rings': 0.0,
                        'heavy_atoms': 0.0, 'ring_count': 0.0, 'formal_charge': 0.0
                    }
                else:
                    # Compute key descriptors for MDM2 binding
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
                        
                        # Additional binding-relevant descriptors
                        'molar_refractivity': Descriptors.MolMR(mol),
                        'balaban_j': Descriptors.BalabanJ(mol),
                        'bertz_ct': Descriptors.BertzCT(mol),
                        'qed': Descriptors.qed(mol),
                        
                        # Lipinski violations
                        'lipinski_violations': sum([
                            Descriptors.MolWt(mol) > 500,
                            Descriptors.MolLogP(mol) > 5,
                            Descriptors.NumHAcceptors(mol) > 10,
                            Descriptors.NumHDonors(mol) > 5
                        ]),
                        
                        # Atom counts (important for binding specificity)
                        'num_carbon': sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6),
                        'num_nitrogen': sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7),
                        'num_oxygen': sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8),
                        'num_sulfur': sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 16),
                        'num_halogen': sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in [9, 17, 35, 53]),
                    }
                
                # Add SMILES for tracking
                desc_dict['smiles'] = smiles
                descriptors_data.append(desc_dict)
                
            except Exception as e:
                logger.warning(f"Error processing SMILES {smiles}: {e}")
                continue
        
        return pd.DataFrame(descriptors_data)
    
    def train_ensemble(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
        """Train ensemble with rigorous cross-validation"""
        logger.info("Training research-grade ensemble...")
        
        # Compute descriptors
        train_descriptors = self.compute_molecular_descriptors(train_df['canonical_smiles'].tolist())
        val_descriptors = self.compute_molecular_descriptors(val_df['canonical_smiles'].tolist())
        
        if train_descriptors.empty or val_descriptors.empty:
            raise ValueError("Failed to compute descriptors")
        
        # Prepare feature matrices
        feature_columns = [col for col in train_descriptors.columns if col != 'smiles']
        
        X_train = train_descriptors[feature_columns].fillna(0)
        y_train = train_df['is_inhibitor'].values
        
        X_val = val_descriptors[feature_columns].fillna(0)
        y_val = val_df['is_inhibitor'].values
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        logger.info(f"Feature matrix: {X_train_scaled.shape}")
        logger.info(f"Class distribution: {np.bincount(y_train)}")
        
        # Train individual models with calibration
        training_results = {}
        
        for model_name, base_model in self.base_models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Cross-validation evaluation
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = []
                
                for fold, (train_idx, test_idx) in enumerate(cv.split(X_train_scaled, y_train)):
                    X_fold_train, X_fold_test = X_train_scaled[train_idx], X_train_scaled[test_idx]
                    y_fold_train, y_fold_test = y_train[train_idx], y_train[test_idx]
                    
                    # Train model
                    fold_model = base_model.__class__(**base_model.get_params())
                    fold_model.fit(X_fold_train, y_fold_train)
                    
                    # Evaluate
                    fold_pred = fold_model.predict(X_fold_test)
                    fold_f1 = f1_score(y_fold_test, fold_pred)
                    cv_scores.append(fold_f1)
                
                # Train calibrated model on full training set
                calibrated_model = CalibratedClassifierCV(
                    base_model, method='isotonic', cv=5
                )
                calibrated_model.fit(X_train_scaled, y_train)
                
                # Evaluate on validation set
                val_proba = calibrated_model.predict_proba(X_val_scaled)[:, 1]
                val_pred = (val_proba >= 0.5).astype(int)
                
                val_metrics = self._calculate_metrics(y_val, val_pred, val_proba)
                
                # Store model and results
                self.calibrated_models[model_name] = calibrated_model
                training_results[model_name] = {
                    'cv_f1_mean': np.mean(cv_scores),
                    'cv_f1_std': np.std(cv_scores),
                    'val_metrics': val_metrics
                }
                
                logger.info(f"{model_name} - CV F1: {np.mean(cv_scores):.3f}±{np.std(cv_scores):.3f}, "
                          f"Val F1: {val_metrics['f1_score']:.3f}")
                
            except Exception as e:
                logger.warning(f"Failed to train {model_name}: {e}")
                continue
        
        # Optimize ensemble threshold
        self.optimal_threshold = self._optimize_ensemble_threshold(X_val_scaled, y_val)
        
        # Final ensemble evaluation
        final_metrics = self._evaluate_ensemble(X_val_scaled, y_val)
        
        logger.info(f"Ensemble performance - F1: {final_metrics['f1_score']:.3f}, "
                   f"Specificity: {final_metrics.get('specificity', 'N/A'):.3f}")
        
        return {
            'individual_results': training_results,
            'ensemble_metrics': final_metrics,
            'optimal_threshold': self.optimal_threshold,
            'feature_names': feature_columns
        }
    
    def _optimize_ensemble_threshold(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Optimize ensemble threshold for target specificity"""
        logger.info("Optimizing ensemble threshold...")
        
        # Get ensemble probabilities
        ensemble_proba = self._get_ensemble_probabilities(X_val)
        
        best_threshold = 0.5
        best_score = 0.0
        
        thresholds = np.arange(0.1, 0.95, 0.02)
        
        for threshold in thresholds:
            pred = (ensemble_proba >= threshold).astype(int)
            
            cm = confusion_matrix(y_val, pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                # Score prioritizing specificity
                if specificity >= self.specificity_target:
                    score = specificity + 0.3 * sensitivity
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
        
        logger.info(f"Optimal threshold: {best_threshold:.3f}")
        return best_threshold
    
    def _get_ensemble_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble probabilities"""
        ensemble_proba = np.zeros(len(X))
        
        for model in self.calibrated_models.values():
            try:
                proba = model.predict_proba(X)[:, 1]
                ensemble_proba += proba
            except:
                continue
        
        return ensemble_proba / len(self.calibrated_models)
    
    def _evaluate_ensemble(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate ensemble performance"""
        ensemble_proba = self._get_ensemble_probabilities(X)
        ensemble_pred = (ensemble_proba >= self.optimal_threshold).astype(int)
        
        return self._calculate_metrics(y, ensemble_pred, ensemble_proba)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict:
        """Calculate comprehensive metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # Add AUC if we have both classes
        if len(set(y_true)) > 1:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
        
        # Calculate specificity and sensitivity
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        return metrics
    
    def external_validation(self, external_smiles: List[str], 
                          expected_labels: Optional[List[int]] = None) -> List[Dict]:
        """Perform external validation on independent compounds"""
        logger.info(f"External validation on {len(external_smiles)} compounds...")
        
        # Compute descriptors
        descriptors = self.compute_molecular_descriptors(external_smiles)
        
        if descriptors.empty:
            return []
        
        # Prepare features
        feature_columns = [col for col in descriptors.columns if col != 'smiles']
        X = descriptors[feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        ensemble_proba = self._get_ensemble_probabilities(X_scaled)
        ensemble_pred = (ensemble_proba >= self.optimal_threshold).astype(int)
        
        # Individual model predictions for analysis
        individual_predictions = {}
        for model_name, model in self.calibrated_models.items():
            try:
                proba = model.predict_proba(X_scaled)[:, 1]
                individual_predictions[model_name] = proba
            except Exception as e:
                logger.warning(f"Error with model {model_name}: {e}")
        
        # Format results
        results = []
        for i, smiles in enumerate(external_smiles):
            result = {
                'smiles': smiles,
                'ensemble_prediction': 'Inhibitor' if ensemble_pred[i] == 1 else 'Non-inhibitor',
                'ensemble_probability': float(ensemble_proba[i]),
                'binary_prediction': int(ensemble_pred[i]),
                'confidence': 'High' if abs(ensemble_proba[i] - 0.5) > 0.3 else 
                             'Medium' if abs(ensemble_proba[i] - 0.5) > 0.15 else 'Low',
                'individual_models': {name: float(pred[i]) for name, pred in individual_predictions.items()}
            }
            
            # Add expected label if provided
            if expected_labels and i < len(expected_labels):
                result['expected_label'] = expected_labels[i]
                result['correct_prediction'] = (ensemble_pred[i] == expected_labels[i])
            
            results.append(result)
        
        # Calculate external validation metrics if expected labels provided
        if expected_labels:
            external_metrics = self._calculate_metrics(
                np.array(expected_labels), ensemble_pred, ensemble_proba
            )
            
            logger.info("External Validation Metrics:")
            for metric, value in external_metrics.items():
                logger.info(f"  {metric}: {value:.3f}")
        
        return results
    
    def predict(self, smiles_list: List[str]) -> List[Dict]:
        """Make predictions on new SMILES"""
        return self.external_validation(smiles_list)
    
    def save_model(self, filepath: str) -> None:
        """Save trained ensemble model"""
        model_data = {
            'calibrated_models': self.calibrated_models,
            'scaler': self.scaler,
            'optimal_threshold': self.optimal_threshold,
            'specificity_target': self.specificity_target
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Ensemble model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'ResearchGradeEnsemble':
        """Load saved ensemble model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        ensemble = cls(specificity_target=model_data['specificity_target'])
        ensemble.calibrated_models = model_data['calibrated_models']
        ensemble.scaler = model_data['scaler']
        ensemble.optimal_threshold = model_data['optimal_threshold']
        
        return ensemble

def main():
    """Train and test research-grade ensemble"""
    # Load rigorous dataset
    train_df = pd.read_csv("data/rigorous/train_rigorous.csv")
    val_df = pd.read_csv("data/rigorous/val_rigorous.csv")
    test_df = pd.read_csv("data/rigorous/test_rigorous.csv")
    
    logger.info(f"Training: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
    # Initialize ensemble
    ensemble = ResearchGradeEnsemble(specificity_target=0.90)
    
    # Train ensemble
    results = ensemble.train_ensemble(train_df, val_df)
    
    # External validation on test set
    test_smiles = test_df['canonical_smiles'].tolist()
    test_labels = test_df['is_inhibitor'].tolist()
    
    external_results = ensemble.external_validation(test_smiles, test_labels)
    
    # Save model
    ensemble.save_model("models/research_grade_ensemble.pkl")
    
    # Test on known compounds
    known_inhibitors = [
        "CC1=C(C=C(C=C1)C(=O)N2CCN(CC2)C(=O)C3=CC=C(C=C3)F)NC4=NC=C(C=N4)C5=CN=CC=C5",  # Milademetan
        "CC1=C(C(=NO1)C2=CC=C(C=C2)C(F)(F)F)C(=O)N3CCC(CC3)C4=NC=CN4"  # Navtemadlin
    ]
    
    known_non_inhibitors = [
        "CCO",  # Ethanol
        "CC(=O)NC1=CC=C(C=C1)O",  # Acetaminophen
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    ]
    
    inhibitor_predictions = ensemble.predict(known_inhibitors)
    non_inhibitor_predictions = ensemble.predict(known_non_inhibitors)
    
    # Print results
    print("\n" + "="*80)
    print("RESEARCH-GRADE ENSEMBLE RESULTS")
    print("="*80)
    
    print(f"\nTraining Summary:")
    print(f"  Optimal threshold: {results['optimal_threshold']:.3f}")
    print(f"  Feature count: {len(results['feature_names'])}")
    
    print(f"\nEnsemble Performance:")
    for metric, value in results['ensemble_metrics'].items():
        print(f"  {metric}: {value:.3f}")
    
    print(f"\nIndividual Model Performance:")
    for model_name, model_results in results['individual_results'].items():
        val_metrics = model_results['val_metrics']
        print(f"  {model_name}:")
        print(f"    CV F1: {model_results['cv_f1_mean']:.3f}±{model_results['cv_f1_std']:.3f}")
        print(f"    Val F1: {val_metrics['f1_score']:.3f}")
        print(f"    Val Specificity: {val_metrics.get('specificity', 'N/A'):.3f}")
    
    # Count correct external predictions
    correct_predictions = sum(1 for result in external_results if result.get('correct_prediction', False))
    total_predictions = len(external_results)
    external_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    print(f"\nExternal Validation (Test Set):")
    print(f"  Accuracy: {external_accuracy:.3f} ({correct_predictions}/{total_predictions})")
    
    print(f"\nKnown Inhibitor Predictions:")
    for pred in inhibitor_predictions:
        print(f"  {pred['smiles'][:50]}...")
        print(f"    Prediction: {pred['ensemble_prediction']}")
        print(f"    Probability: {pred['ensemble_probability']:.3f}")
        print(f"    Confidence: {pred['confidence']}")
    
    print(f"\nKnown Non-inhibitor Predictions:")
    for pred in non_inhibitor_predictions:
        print(f"  {pred['smiles']}")
        print(f"    Prediction: {pred['ensemble_prediction']}")
        print(f"    Probability: {pred['ensemble_probability']:.3f}")
        print(f"    Confidence: {pred['confidence']}")
    
    print(f"\n✅ Research-grade ensemble training completed!")
    print(f"💾 Model saved to models/research_grade_ensemble.pkl")

if __name__ == "__main__":
    main()