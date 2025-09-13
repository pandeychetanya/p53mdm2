"""
Hard Negative Mining Pipeline for MDM2 Inhibitor Prediction
Implements iterative training with hard negative mining to improve specificity
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import logging
from typing import List, Dict, Tuple, Optional, Union
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HardNegativeMiner:
    """Implements hard negative mining for improved model specificity"""
    
    def __init__(self, base_models: Optional[Dict] = None, confidence_threshold: float = 0.7):
        """
        Initialize hard negative miner
        
        Args:
            base_models: Dictionary of models to use for ensemble
            confidence_threshold: Threshold for considering a negative prediction as "hard"
        """
        self.confidence_threshold = confidence_threshold
        self.scaler = StandardScaler()
        
        # Initialize base models
        if base_models is None:
            self.base_models = {
                'random_forest': RandomForestClassifier(
                    n_estimators=200, max_depth=10, min_samples_split=10,
                    min_samples_leaf=5, random_state=42, n_jobs=-1
                ),
                'logistic_regression': LogisticRegression(
                    random_state=42, max_iter=1000, class_weight='balanced'
                ),
                'svm': SVC(
                    kernel='rbf', probability=True, random_state=42, class_weight='balanced'
                ),
                'xgboost': xgb.XGBClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    random_state=42, eval_metric='logloss'
                )
            }
        else:
            self.base_models = base_models
        
        self.calibrated_models = {}
        self.hard_negatives_history = []
        self.mining_stats = []
        
    def compute_molecular_descriptors(self, smiles_list: List[str]) -> pd.DataFrame:
        """Compute molecular descriptors for SMILES strings"""
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
                    # Compute robust descriptors
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
                        
                        # Atom counts
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
    
    def identify_hard_negatives(self, X: np.ndarray, y: np.ndarray, 
                              smiles: List[str]) -> Tuple[List[str], List[float]]:
        """Identify hard negative examples (non-inhibitors predicted as inhibitors)"""
        hard_negatives = []
        hard_negative_scores = []
        
        # Get non-inhibitor indices
        negative_indices = np.where(y == 0)[0]
        
        if len(negative_indices) == 0:
            return hard_negatives, hard_negative_scores
        
        # Predict on negatives using ensemble
        ensemble_predictions = np.zeros(len(negative_indices))
        
        for model_name, model in self.calibrated_models.items():
            try:
                neg_proba = model.predict_proba(X[negative_indices])[:, 1]
                ensemble_predictions += neg_proba
            except Exception as e:
                logger.warning(f"Error with model {model_name}: {e}")
                continue
        
        ensemble_predictions /= len(self.calibrated_models)
        
        # Find hard negatives (high confidence incorrect predictions)
        hard_indices = negative_indices[ensemble_predictions >= self.confidence_threshold]
        
        for idx in hard_indices:
            hard_negatives.append(smiles[idx])
            hard_negative_scores.append(ensemble_predictions[np.where(negative_indices == idx)[0][0]])
        
        return hard_negatives, hard_negative_scores
    
    def mine_external_hard_negatives(self, external_smiles: List[str], 
                                   batch_size: int = 100) -> List[Tuple[str, float]]:
        """Mine hard negatives from external compound libraries"""
        logger.info(f"Mining hard negatives from {len(external_smiles)} external compounds...")
        
        hard_negatives = []
        
        # Process in batches to manage memory
        for i in range(0, len(external_smiles), batch_size):
            batch_smiles = external_smiles[i:i+batch_size]
            
            # Compute descriptors for batch
            batch_descriptors = self.compute_molecular_descriptors(batch_smiles)
            
            if batch_descriptors.empty:
                continue
            
            # Prepare features (excluding smiles column)
            feature_columns = [col for col in batch_descriptors.columns if col != 'smiles']
            X_batch = batch_descriptors[feature_columns].fillna(0)
            
            # Scale features
            X_batch_scaled = self.scaler.transform(X_batch)
            
            # Get ensemble predictions
            batch_predictions = np.zeros(len(X_batch_scaled))
            
            for model_name, model in self.calibrated_models.items():
                try:
                    batch_proba = model.predict_proba(X_batch_scaled)[:, 1]
                    batch_predictions += batch_proba
                except Exception as e:
                    logger.warning(f"Error with model {model_name} on batch: {e}")
                    continue
            
            batch_predictions /= len(self.calibrated_models)
            
            # Find hard negatives in this batch
            hard_indices = np.where(batch_predictions >= self.confidence_threshold)[0]
            
            for idx in hard_indices:
                original_idx = i + idx
                if original_idx < len(batch_smiles):
                    hard_negatives.append((
                        batch_smiles[idx], 
                        float(batch_predictions[idx])
                    ))
        
        logger.info(f"Found {len(hard_negatives)} hard negatives from external mining")
        return hard_negatives
    
    def train_with_hard_negatives(self, train_df: pd.DataFrame, 
                                val_df: pd.DataFrame,
                                external_compounds: Optional[List[str]] = None,
                                max_iterations: int = 3) -> Dict:
        """Train models with iterative hard negative mining"""
        logger.info("Starting hard negative mining training...")\n        \n        # Compute descriptors for training and validation\n        train_descriptors = self.compute_molecular_descriptors(train_df['canonical_smiles'].tolist())\n        val_descriptors = self.compute_molecular_descriptors(val_df['canonical_smiles'].tolist())\n        \n        if train_descriptors.empty or val_descriptors.empty:\n            raise ValueError(\"Failed to compute descriptors\")\n        \n        # Prepare feature matrices\n        feature_columns = [col for col in train_descriptors.columns if col != 'smiles']\n        \n        X_train = train_descriptors[feature_columns].fillna(0)\n        y_train = train_df['is_inhibitor'].values\n        \n        X_val = val_descriptors[feature_columns].fillna(0)\n        y_val = val_df['is_inhibitor'].values\n        \n        # Scale features\n        X_train_scaled = self.scaler.fit_transform(X_train)\n        X_val_scaled = self.scaler.transform(X_val)\n        \n        # Store original training data\n        original_X_train = X_train_scaled.copy()\n        original_y_train = y_train.copy()\n        original_smiles = train_df['canonical_smiles'].tolist()\n        \n        training_history = []\n        \n        for iteration in range(max_iterations):\n            logger.info(f\"Hard negative mining iteration {iteration + 1}/{max_iterations}\")\n            \n            # Train base models with calibration\n            iteration_models = {}\n            \n            for model_name, base_model in self.base_models.items():\n                logger.info(f\"Training {model_name}...\")\n                \n                try:\n                    # Use calibrated classifier\n                    calibrated_model = CalibratedClassifierCV(\n                        base_model, method='isotonic', cv=3\n                    )\n                    calibrated_model.fit(X_train_scaled, y_train)\n                    iteration_models[model_name] = calibrated_model\n                    \n                except Exception as e:\n                    logger.warning(f\"Failed to train {model_name}: {e}\")\n                    continue\n            \n            self.calibrated_models = iteration_models\n            \n            # Evaluate current models\n            val_metrics = self._evaluate_ensemble(X_val_scaled, y_val)\n            \n            # Mine hard negatives from current training set\n            hard_negatives, hard_scores = self.identify_hard_negatives(\n                X_train_scaled, y_train, train_df['canonical_smiles'].tolist()\n            )\n            \n            # Mine from external compounds if available\n            external_hard_negatives = []\n            if external_compounds and len(external_compounds) > 0:\n                external_hard_negatives = self.mine_external_hard_negatives(\n                    external_compounds, batch_size=50\n                )\n            \n            # Update training set with hard negatives\n            if hard_negatives or external_hard_negatives:\n                # Add internal hard negatives\n                for smiles in hard_negatives:\n                    if smiles not in original_smiles:  # Avoid duplicates\n                        new_descriptors = self.compute_molecular_descriptors([smiles])\n                        if not new_descriptors.empty:\n                            X_new = new_descriptors[feature_columns].fillna(0)\n                            X_new_scaled = self.scaler.transform(X_new)\n                            \n                            X_train_scaled = np.vstack([X_train_scaled, X_new_scaled])\n                            y_train = np.append(y_train, 0)  # Hard negative\n                \n                # Add external hard negatives\n                for smiles, score in external_hard_negatives:\n                    new_descriptors = self.compute_molecular_descriptors([smiles])\n                    if not new_descriptors.empty:\n                        X_new = new_descriptors[feature_columns].fillna(0)\n                        X_new_scaled = self.scaler.transform(X_new)\n                        \n                        X_train_scaled = np.vstack([X_train_scaled, X_new_scaled])\n                        y_train = np.append(y_train, 0)  # External hard negative\n            \n            # Store iteration statistics\n            iteration_stats = {\n                'iteration': iteration + 1,\n                'training_size': len(y_train),\n                'hard_negatives_found': len(hard_negatives),\n                'external_hard_negatives': len(external_hard_negatives),\n                'validation_metrics': val_metrics\n            }\n            \n            self.mining_stats.append(iteration_stats)\n            training_history.append(iteration_stats)\n            \n            logger.info(f\"Iteration {iteration + 1} stats:\")\n            logger.info(f\"  Training set size: {len(y_train)}\")\n            logger.info(f\"  Hard negatives found: {len(hard_negatives)}\")\n            logger.info(f\"  External hard negatives: {len(external_hard_negatives)}\")\n            logger.info(f\"  Validation F1: {val_metrics['f1_score']:.3f}\")\n            logger.info(f\"  Validation Specificity: {val_metrics.get('specificity', 'N/A')}\")\n            \n            # Early stopping if no improvement\n            if iteration > 0:\n                prev_f1 = training_history[iteration-1]['validation_metrics']['f1_score']\n                current_f1 = val_metrics['f1_score']\n                \n                if current_f1 <= prev_f1 and len(hard_negatives) + len(external_hard_negatives) < 5:\n                    logger.info(\"Early stopping: no improvement and few hard negatives found\")\n                    break\n        \n        return {\n            'final_models': self.calibrated_models,\n            'training_history': training_history,\n            'final_training_size': len(y_train),\n            'scaler': self.scaler\n        }\n    \n    def _evaluate_ensemble(self, X: np.ndarray, y: np.ndarray) -> Dict:\n        \"\"\"Evaluate ensemble model performance\"\"\"\n        if not self.calibrated_models:\n            return {}\n        \n        # Get ensemble predictions\n        ensemble_proba = np.zeros(len(y))\n        \n        for model in self.calibrated_models.values():\n            try:\n                proba = model.predict_proba(X)[:, 1]\n                ensemble_proba += proba\n            except:\n                continue\n        \n        ensemble_proba /= len(self.calibrated_models)\n        ensemble_pred = (ensemble_proba >= 0.5).astype(int)\n        \n        # Calculate metrics\n        metrics = {\n            'accuracy': accuracy_score(y, ensemble_pred),\n            'precision': precision_score(y, ensemble_pred, zero_division=0),\n            'recall': recall_score(y, ensemble_pred, zero_division=0),\n            'f1_score': f1_score(y, ensemble_pred, zero_division=0),\n        }\n        \n        # Add AUC if we have both classes\n        if len(set(y)) > 1:\n            metrics['auc_roc'] = roc_auc_score(y, ensemble_proba)\n        \n        # Calculate specificity and sensitivity\n        from sklearn.metrics import confusion_matrix\n        cm = confusion_matrix(y, ensemble_pred)\n        if cm.shape == (2, 2):\n            tn, fp, fn, tp = cm.ravel()\n            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0\n            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0\n        \n        return metrics\n    \n    def predict_ensemble(self, smiles_list: List[str]) -> List[Dict]:\n        \"\"\"Make ensemble predictions on new SMILES\"\"\"\n        # Compute descriptors\n        descriptors = self.compute_molecular_descriptors(smiles_list)\n        \n        if descriptors.empty:\n            return []\n        \n        # Prepare features\n        feature_columns = [col for col in descriptors.columns if col != 'smiles']\n        X = descriptors[feature_columns].fillna(0)\n        X_scaled = self.scaler.transform(X)\n        \n        # Get ensemble predictions\n        ensemble_proba = np.zeros(len(X_scaled))\n        model_predictions = {}\n        \n        for model_name, model in self.calibrated_models.items():\n            try:\n                proba = model.predict_proba(X_scaled)[:, 1]\n                ensemble_proba += proba\n                model_predictions[model_name] = proba\n            except Exception as e:\n                logger.warning(f\"Error with model {model_name}: {e}\")\n                continue\n        \n        ensemble_proba /= len(self.calibrated_models)\n        ensemble_pred = (ensemble_proba >= 0.5).astype(int)\n        \n        # Format results\n        results = []\n        for i, smiles in enumerate(smiles_list):\n            result = {\n                'smiles': smiles,\n                'prediction': 'Inhibitor' if ensemble_pred[i] == 1 else 'Non-inhibitor',\n                'ensemble_probability': float(ensemble_proba[i]),\n                'confidence': 'High' if abs(ensemble_proba[i] - 0.5) > 0.3 else \n                             'Medium' if abs(ensemble_proba[i] - 0.5) > 0.15 else 'Low',\n                'model_predictions': {name: float(pred[i]) for name, pred in model_predictions.items()}\n            }\n            results.append(result)\n        \n        return results\n    \n    def save_models(self, filepath: str) -> None:\n        \"\"\"Save trained models and components\"\"\"\n        model_data = {\n            'calibrated_models': self.calibrated_models,\n            'scaler': self.scaler,\n            'mining_stats': self.mining_stats,\n            'confidence_threshold': self.confidence_threshold\n        }\n        \n        with open(filepath, 'wb') as f:\n            pickle.dump(model_data, f)\n        \n        logger.info(f\"Models saved to {filepath}\")\n    \n    @classmethod\n    def load_models(cls, filepath: str) -> 'HardNegativeMiner':\n        \"\"\"Load saved models\"\"\"\n        with open(filepath, 'rb') as f:\n            model_data = pickle.load(f)\n        \n        miner = cls(confidence_threshold=model_data['confidence_threshold'])\n        miner.calibrated_models = model_data['calibrated_models']\n        miner.scaler = model_data['scaler']\n        miner.mining_stats = model_data['mining_stats']\n        \n        return miner

def main():
    \"\"\"Test hard negative mining pipeline\"\"\"
    # Load rigorous dataset\n    train_df = pd.read_csv(\"data/rigorous/train_rigorous.csv\")\n    val_df = pd.read_csv(\"data/rigorous/val_rigorous.csv\")\n    \n    logger.info(f\"Training set: {len(train_df)} compounds\")\n    logger.info(f\"Validation set: {len(val_df)} compounds\")\n    \n    # Initialize hard negative miner\n    miner = HardNegativeMiner(confidence_threshold=0.6)\n    \n    # Create external compound library (drug-like non-inhibitors)\n    external_compounds = [\n        \"CC(=O)NC1=CC=C(C=C1)O\",  # Acetaminophen\n        \"CC(C)CC1=CC=C(C=C1)C(C)C(=O)O\",  # Ibuprofen\n        \"CC(=O)OC1=CC=CC=C1C(=O)O\",  # Aspirin\n        \"CN1C=NC2=C1C(=O)N(C(=O)N2C)C\",  # Caffeine\n        \"CN(C)C(=N)N=C(N)N\",  # Metformin\n        \"CC1=CC=C(C=C1)C(C)C(=O)O\",  # Naproxen\n        \"CC1=CC=C(C=C1)S(=O)(=O)N\",  # Tolbutamide-like\n        \"CCCCC1=CC=CC=C1C(=O)O\",  # Valproic acid-like\n        \"CC(C)(C)C(=O)O\",  # Pivalic acid\n        \"CC1=CC=CC=C1NC(=O)C\"  # Acetanilide-like\n    ]\n    \n    # Train with hard negative mining\n    results = miner.train_with_hard_negatives(\n        train_df, val_df, \n        external_compounds=external_compounds,\n        max_iterations=2  # Reduced for testing\n    )\n    \n    # Save models\n    miner.save_models(\"models/hard_negative_miner.pkl\")\n    \n    # Test predictions\n    test_smiles = [\n        \"CCO\",  # Ethanol\n        \"CC(=O)NC1=CC=C(C=C1)O\",  # Acetaminophen\n        \"CN1C=NC2=C1C(=O)N(C(=O)N2C)C\",  # Caffeine\n    ]\n    \n    predictions = miner.predict_ensemble(test_smiles)\n    \n    # Print results\n    print(\"\\n\" + \"=\"*80)\n    print(\"HARD NEGATIVE MINING RESULTS\")\n    print(\"=\"*80)\n    \n    print(f\"\\nTraining Summary:\")\n    print(f\"  Iterations completed: {len(results['training_history'])}\")\n    print(f\"  Final training set size: {results['final_training_size']}\")\n    \n    for i, stats in enumerate(results['training_history']):\n        print(f\"\\n  Iteration {i+1}:\")\n        print(f\"    Hard negatives found: {stats['hard_negatives_found']}\")\n        print(f\"    External hard negatives: {stats['external_hard_negatives']}\")\n        print(f\"    Validation F1: {stats['validation_metrics']['f1_score']:.3f}\")\n        if 'specificity' in stats['validation_metrics']:\n            print(f\"    Validation Specificity: {stats['validation_metrics']['specificity']:.3f}\")\n    \n    print(f\"\\nTest Predictions:\")\n    for pred in predictions:\n        print(f\"  {pred['smiles']}\")\n        print(f\"    Prediction: {pred['prediction']}\")\n        print(f\"    Ensemble Probability: {pred['ensemble_probability']:.3f}\")\n        print(f\"    Confidence: {pred['confidence']}\")\n        print()\n    \n    print(f\"✅ Hard negative mining completed!\")\n    print(f\"💾 Models saved to models/hard_negative_miner.pkl\")\n\nif __name__ == \"__main__\":\n    main()