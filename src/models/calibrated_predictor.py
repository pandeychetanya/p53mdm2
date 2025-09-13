"""
Calibrated MDM2 Inhibitor Predictor with Improved Probability Estimates
Combines balanced data, optimized thresholds, and probability calibration for production use
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.isotonic import IsotonicRegression
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CalibratedMDM2Predictor:
    """Production-ready MDM2 inhibitor predictor with calibrated probabilities"""
    
    def __init__(self, specificity_target: float = 0.90):
        """
        Initialize calibrated predictor
        
        Args:
            specificity_target: Target specificity for threshold optimization
        """
        self.specificity_target = specificity_target
        self.base_model = None
        self.calibrated_model = None
        self.scaler = StandardScaler()
        self.optimal_threshold = 0.5
        self.feature_names = None
        self.calibration_method = 'isotonic'  # 'isotonic' or 'sigmoid'
        
    def load_and_prepare_data(self, balanced_data_path: str, test_data_path: str) -> Tuple:
        """Load and prepare training and test data"""
        logger.info("Loading and preparing data...")
        
        # Load balanced training data
        train_df = pd.read_csv(balanced_data_path)
        
        # Use optimized features (from SHAP analysis)
        self.feature_names = ['alogp', 'hba', 'aromatic_rings', 'cx_logp']
        
        # Prepare training features
        X_train = train_df[self.feature_names].fillna(0)
        y_train = train_df['is_inhibitor'].values
        
        # Load specificity test data
        test_df = pd.read_csv(test_data_path)
        X_test = self._compute_descriptors_from_smiles(test_df['canonical_smiles'].tolist())
        y_test = test_df['is_inhibitor'].values
        
        logger.info(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
        logger.info(f"Training class distribution: {np.bincount(y_train)}")
        
        return X_train, y_train, X_test, y_test
    
    def _compute_descriptors_from_smiles(self, smiles_list: List[str]) -> np.ndarray:
        """Compute molecular descriptors from SMILES strings"""
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        
        descriptors_data = []
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    descriptors_data.append([0.0] * len(self.feature_names))
                    continue
                
                desc_values = []
                for feature in self.feature_names:
                    if feature == 'alogp':
                        desc_values.append(Descriptors.MolLogP(mol))
                    elif feature == 'hba':
                        desc_values.append(Descriptors.NumHAcceptors(mol))
                    elif feature == 'aromatic_rings':
                        desc_values.append(Descriptors.NumAromaticRings(mol))
                    elif feature == 'cx_logp':
                        desc_values.append(Descriptors.MolLogP(mol))  # Use MolLogP as proxy
                    else:
                        desc_values.append(0.0)
                
                descriptors_data.append(desc_values)
                
            except Exception as e:
                logger.warning(f"Error processing SMILES {smiles}: {e}")
                descriptors_data.append([0.0] * len(self.feature_names))
        
        return np.array(descriptors_data)
    
    def train_calibrated_model(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_test: Optional[np.ndarray] = None, 
                             y_test: Optional[np.ndarray] = None) -> Dict:
        """Train base model and apply probability calibration"""
        logger.info("Training calibrated model...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train base model with optimized parameters
        self.base_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight={0: 1, 1: 2.5},  # Higher weight for inhibitors
            random_state=42,
            bootstrap=True,
            max_features='sqrt'
        )
        
        # Use stratified cross-validation for calibration
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Create calibrated classifier
        logger.info(f"Applying {self.calibration_method} calibration...")
        self.calibrated_model = CalibratedClassifierCV(
            self.base_model,
            method=self.calibration_method,
            cv=cv_strategy,
            n_jobs=-1
        )
        
        # First fit base model separately to get uncalibrated predictions
        self.base_model.fit(X_train_scaled, y_train)
        
        # Fit calibrated model
        self.calibrated_model.fit(X_train_scaled, y_train)
        
        # Optimize threshold using calibrated probabilities
        calibrated_proba = self.calibrated_model.predict_proba(X_train_scaled)[:, 1]
        self.optimal_threshold = self._optimize_threshold_for_specificity(y_train, calibrated_proba)
        
        # Evaluate calibration quality
        results = self._evaluate_calibration(X_train_scaled, y_train, X_test, y_test)
        results['optimal_threshold'] = self.optimal_threshold
        results['feature_names'] = self.feature_names
        
        logger.info(f"Training completed. Optimal threshold: {self.optimal_threshold:.3f}")
        
        return results
    
    def _optimize_threshold_for_specificity(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Optimize threshold to achieve target specificity"""
        thresholds = np.arange(0.1, 0.95, 0.02)
        best_threshold = 0.5
        best_score = 0.0
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            # Calculate confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred)
            
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                # Prioritize specificity but maintain reasonable sensitivity
                if specificity >= self.specificity_target and sensitivity >= 0.7:
                    score = specificity + 0.3 * sensitivity
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
        
        return best_threshold
    
    def _evaluate_calibration(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_test: Optional[np.ndarray] = None, 
                            y_test: Optional[np.ndarray] = None) -> Dict:
        """Evaluate probability calibration quality"""
        results = {}
        
        # Training set evaluation
        train_proba_uncal = self.base_model.predict_proba(X_train)[:, 1]
        train_proba_cal = self.calibrated_model.predict_proba(X_train)[:, 1]
        
        results['train_metrics'] = {
            'brier_score_uncalibrated': brier_score_loss(y_train, train_proba_uncal),
            'brier_score_calibrated': brier_score_loss(y_train, train_proba_cal),
            'log_loss_uncalibrated': log_loss(y_train, train_proba_uncal),
            'log_loss_calibrated': log_loss(y_train, train_proba_cal)
        }
        
        # Test set evaluation if provided
        if X_test is not None and y_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            
            test_proba_uncal = self.base_model.predict_proba(X_test_scaled)[:, 1]
            test_proba_cal = self.calibrated_model.predict_proba(X_test_scaled)[:, 1]
            
            results['test_metrics'] = {
                'brier_score_uncalibrated': brier_score_loss(y_test, test_proba_uncal),
                'brier_score_calibrated': brier_score_loss(y_test, test_proba_cal),
                'log_loss_uncalibrated': log_loss(y_test, test_proba_uncal) if len(set(y_test)) > 1 else np.nan,
                'log_loss_calibrated': log_loss(y_test, test_proba_cal) if len(set(y_test)) > 1 else np.nan
            }
            
            # Specificity analysis
            test_pred_cal = (test_proba_cal >= self.optimal_threshold).astype(int)
            false_positives = sum(test_pred_cal)  # All test samples should be non-inhibitors
            
            results['specificity_analysis'] = {
                'total_test_compounds': len(y_test),
                'false_positives': false_positives,
                'true_negatives': len(y_test) - false_positives,
                'specificity': (len(y_test) - false_positives) / len(y_test),
                'test_probabilities': test_proba_cal.tolist()
            }
        
        return results
    
    def predict_smiles(self, smiles_list: Union[str, List[str]], 
                      return_probabilities: bool = True) -> Union[Dict, List[Dict]]:
        """Predict MDM2 inhibition for SMILES string(s)"""
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        
        # Compute descriptors
        X = self._compute_descriptors_from_smiles(smiles_list)
        X_scaled = self.scaler.transform(X)
        
        # Get calibrated probabilities
        probabilities = self.calibrated_model.predict_proba(X_scaled)[:, 1]
        
        # Make predictions using optimized threshold
        predictions = (probabilities >= self.optimal_threshold).astype(int)
        
        results = []
        for i, smiles in enumerate(smiles_list):
            result = {
                'smiles': smiles,
                'prediction': 'Inhibitor' if predictions[i] == 1 else 'Non-inhibitor',
                'binary_prediction': int(predictions[i]),
                'confidence': 'High' if abs(probabilities[i] - 0.5) > 0.3 else 'Medium' if abs(probabilities[i] - 0.5) > 0.15 else 'Low'
            }
            
            if return_probabilities:
                result['inhibitor_probability'] = float(probabilities[i])
                result['non_inhibitor_probability'] = float(1 - probabilities[i])
            
            results.append(result)
        
        return results[0] if len(results) == 1 else results
    
    def create_calibration_plots(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_test: Optional[np.ndarray] = None,
                               y_test: Optional[np.ndarray] = None) -> None:
        """Create calibration diagnostic plots"""
        logger.info("Creating calibration plots...")
        
        import os
        os.makedirs('analysis/calibration_plots', exist_ok=True)
        
        X_train_scaled = self.scaler.transform(X_train)
        
        # Get probabilities
        train_proba_uncal = self.base_model.predict_proba(X_train_scaled)[:, 1]
        train_proba_cal = self.calibrated_model.predict_proba(X_train_scaled)[:, 1]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Calibration curve
        ax = axes[0, 0]
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.8)
        
        # Uncalibrated model
        fraction_pos_uncal, mean_pred_uncal = calibration_curve(y_train, train_proba_uncal, n_bins=10)
        ax.plot(mean_pred_uncal, fraction_pos_uncal, 's-', label='Uncalibrated RF', alpha=0.8)
        
        # Calibrated model
        fraction_pos_cal, mean_pred_cal = calibration_curve(y_train, train_proba_cal, n_bins=10)
        ax.plot(mean_pred_cal, fraction_pos_cal, 'o-', label='Calibrated RF', alpha=0.8)
        
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title('Calibration Curve (Training Data)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Probability histograms
        ax = axes[0, 1]
        
        # Separate by class
        inhibitor_mask = y_train == 1
        non_inhibitor_mask = y_train == 0
        
        ax.hist(train_proba_cal[non_inhibitor_mask], bins=20, alpha=0.6, 
               label='Non-inhibitors', color='red', density=True, range=(0, 1))
        ax.hist(train_proba_cal[inhibitor_mask], bins=20, alpha=0.6, 
               label='Inhibitors', color='blue', density=True, range=(0, 1))
        ax.axvline(x=self.optimal_threshold, color='k', linestyle='--', 
                  label=f'Threshold ({self.optimal_threshold:.3f})')
        
        ax.set_xlabel('Calibrated Probability')
        ax.set_ylabel('Density')
        ax.set_title('Calibrated Probability Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Brier score comparison
        ax = axes[1, 0]
        
        brier_uncal = brier_score_loss(y_train, train_proba_uncal)
        brier_cal = brier_score_loss(y_train, train_proba_cal)
        
        categories = ['Uncalibrated', 'Calibrated']
        scores = [brier_uncal, brier_cal]
        colors = ['red', 'green']
        
        bars = ax.bar(categories, scores, color=colors, alpha=0.7)
        ax.set_ylabel('Brier Score (Lower is Better)')
        ax.set_title('Calibration Quality: Brier Score Comparison')
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                   f'{score:.3f}', ha='center', va='bottom')
        
        # 4. Test set analysis (if available)
        if X_test is not None and y_test is not None:
            ax = axes[1, 1]
            
            X_test_scaled = self.scaler.transform(X_test)
            test_proba_cal = self.calibrated_model.predict_proba(X_test_scaled)[:, 1]
            test_pred_cal = (test_proba_cal >= self.optimal_threshold).astype(int)
            
            # Show test predictions
            false_positives = sum(test_pred_cal)
            true_negatives = len(y_test) - false_positives
            
            ax.bar(['True Negatives\\n(Correct)', 'False Positives\\n(Incorrect)'], 
                  [true_negatives, false_positives],
                  color=['green', 'red'], alpha=0.7)
            
            ax.set_ylabel('Number of Test Compounds')
            ax.set_title('Specificity Test: Calibrated Model')
            
            # Add percentage labels
            total = len(y_test)
            ax.text(0, true_negatives + 0.2, f'{true_negatives/total:.1%}', ha='center', va='bottom')
            if false_positives > 0:
                ax.text(1, false_positives + 0.2, f'{false_positives/total:.1%}', ha='center', va='bottom')
        else:
            axes[1, 1].text(0.5, 0.5, 'No test data provided', ha='center', va='center', 
                           transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_title('Test Set Analysis')
        
        plt.tight_layout()
        plt.savefig('analysis/calibration_plots/calibration_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Calibration plots saved to analysis/calibration_plots/")
    
    def save_model(self, filepath: str) -> None:
        """Save the calibrated model and preprocessing components"""
        model_data = {
            'calibrated_model': self.calibrated_model,
            'scaler': self.scaler,
            'optimal_threshold': self.optimal_threshold,
            'feature_names': self.feature_names,
            'calibration_method': self.calibration_method,
            'specificity_target': self.specificity_target
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'CalibratedMDM2Predictor':
        """Load a saved calibrated model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        predictor = cls()
        predictor.calibrated_model = model_data['calibrated_model']
        predictor.scaler = model_data['scaler']
        predictor.optimal_threshold = model_data['optimal_threshold']
        predictor.feature_names = model_data['feature_names']
        predictor.calibration_method = model_data['calibration_method']
        predictor.specificity_target = model_data['specificity_target']
        
        return predictor

def main():
    """Train and evaluate calibrated MDM2 predictor"""
    predictor = CalibratedMDM2Predictor(specificity_target=0.90)
    
    # Load data
    X_train, y_train, X_test, y_test = predictor.load_and_prepare_data(
        'data/processed/balanced_mdm2_data.csv',
        'data/processed/specificity_test_set.csv'
    )
    
    # Train calibrated model
    results = predictor.train_calibrated_model(X_train, y_train, X_test, y_test)
    
    # Create calibration plots
    predictor.create_calibration_plots(X_train, y_train, X_test, y_test)
    
    # Save model
    predictor.save_model('models/calibrated_mdm2_predictor.pkl')
    
    # Print results
    print("\n" + "="*80)
    print("CALIBRATED MDM2 PREDICTOR RESULTS")
    print("="*80)
    
    print(f"\nModel Configuration:")
    print(f"  Features: {results['feature_names']}")
    print(f"  Optimal Threshold: {results['optimal_threshold']:.3f}")
    print(f"  Calibration Method: {predictor.calibration_method}")
    print(f"  Target Specificity: {predictor.specificity_target:.1%}")
    
    # Training metrics
    train_metrics = results['train_metrics']
    print(f"\nTraining Set Calibration Quality:")
    print(f"  Brier Score (Uncalibrated): {train_metrics['brier_score_uncalibrated']:.4f}")
    print(f"  Brier Score (Calibrated):   {train_metrics['brier_score_calibrated']:.4f}")
    print(f"  Improvement: {((train_metrics['brier_score_uncalibrated'] - train_metrics['brier_score_calibrated']) / train_metrics['brier_score_uncalibrated'] * 100):.1f}%")
    
    # Specificity test
    if 'specificity_analysis' in results:
        spec_analysis = results['specificity_analysis']
        print(f"\nSpecificity Test Results:")
        print(f"  Total Test Compounds: {spec_analysis['total_test_compounds']}")
        print(f"  True Negatives: {spec_analysis['true_negatives']}")
        print(f"  False Positives: {spec_analysis['false_positives']}")
        print(f"  Achieved Specificity: {spec_analysis['specificity']:.1%}")
        
        if spec_analysis['specificity'] >= predictor.specificity_target:
            print(f"  ✅ Target specificity achieved!")
        else:
            print(f"  ⚠️  Target specificity not reached")
    
    # Test predictions on example compounds
    print(f"\nExample Predictions:")
    test_smiles = [
        "CCO",  # Ethanol (should be non-inhibitor)
        "CC(=O)NC1=CC=C(C=C1)O",  # Paracetamol (should be non-inhibitor)
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine (should be non-inhibitor)
    ]
    
    for smiles in test_smiles:
        pred = predictor.predict_smiles(smiles)
        print(f"  {smiles}")
        print(f"    Prediction: {pred['prediction']}")
        print(f"    Probability: {pred['inhibitor_probability']:.3f}")
        print(f"    Confidence: {pred['confidence']}")
        print()
    
    print(f"📊 Calibration plots saved to analysis/calibration_plots/calibration_analysis.png")
    print(f"💾 Model saved to models/calibrated_mdm2_predictor.pkl")

if __name__ == "__main__":
    main()