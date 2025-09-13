"""
Threshold Optimization for MDM2 Inhibitor Prediction
Optimizes prediction thresholds to improve specificity and reduce false positives
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MDM2ThresholdOptimizer:
    """Optimizes prediction thresholds for improved specificity"""
    
    def __init__(self, specificity_target: float = 0.90):
        """
        Initialize threshold optimizer
        
        Args:
            specificity_target: Target specificity (true negative rate)
        """
        self.specificity_target = specificity_target
        self.model = None
        self.scaler = StandardScaler()
        self.optimal_threshold = 0.5
        self.feature_weights = None
        
    def load_data(self, balanced_data_path: str, specificity_test_path: str, 
                  problematic_features: Optional[List[str]] = None) -> Tuple:
        """Load training and specificity test data"""
        logger.info("Loading datasets...")
        
        # Load balanced training data
        train_df = pd.read_csv(balanced_data_path)
        
        # Use recommended features (excluding problematic ones)
        recommended_features = ['alogp', 'hba', 'aromatic_rings', 'cx_logp']  # Exclude hbd, num_ro5_violations, rtb
        
        if problematic_features:
            recommended_features = [f for f in recommended_features if f not in problematic_features]
        
        logger.info(f"Using features: {recommended_features}")
        
        # Prepare training data
        X_train = train_df[recommended_features].fillna(0)
        y_train = train_df['is_inhibitor'].values
        
        # Load specificity test set
        test_df = pd.read_csv(specificity_test_path)
        
        # For test set, we need to compute molecular descriptors
        X_test = self._compute_test_descriptors(test_df['canonical_smiles'].tolist(), recommended_features)
        y_test = test_df['is_inhibitor'].values  # Should all be 0
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"Training labels: {np.bincount(y_train)}, Test labels: {np.bincount(y_test)}")
        
        return X_train, y_train, X_test, y_test, recommended_features
    
    def _compute_test_descriptors(self, smiles_list: List[str], feature_names: List[str]) -> np.ndarray:
        """Compute descriptors for test SMILES"""
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        
        descriptors_data = []
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    # Fill with zeros for invalid SMILES
                    descriptors_data.append([0.0] * len(feature_names))
                    continue
                
                # Compute descriptors
                desc_values = []
                for feature in feature_names:
                    if feature == 'alogp':
                        desc_values.append(Descriptors.MolLogP(mol))
                    elif feature == 'hba':
                        desc_values.append(Descriptors.NumHAcceptors(mol))
                    elif feature == 'hbd':
                        desc_values.append(Descriptors.NumHDonors(mol))
                    elif feature == 'aromatic_rings':
                        desc_values.append(Descriptors.NumAromaticRings(mol))
                    elif feature == 'rtb':
                        desc_values.append(Descriptors.NumRotatableBonds(mol))
                    elif feature == 'num_ro5_violations':
                        # Calculate Lipinski violations
                        mw = Descriptors.MolWt(mol)
                        logp = Descriptors.MolLogP(mol)
                        hba = Descriptors.NumHAcceptors(mol)
                        hbd = Descriptors.NumHDonors(mol)
                        violations = sum([mw > 500, logp > 5, hba > 10, hbd > 5])
                        desc_values.append(violations)
                    elif feature == 'cx_logp':
                        # Use MolLogP as approximation for cx_logp
                        desc_values.append(Descriptors.MolLogP(mol))
                    else:
                        desc_values.append(0.0)
                
                descriptors_data.append(desc_values)
                
            except Exception as e:
                logger.warning(f"Error computing descriptors for {smiles}: {e}")
                descriptors_data.append([0.0] * len(feature_names))
        
        return np.array(descriptors_data)
    
    def optimize_threshold(self, X_train: np.ndarray, y_train: np.ndarray, 
                          X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Optimize prediction threshold for improved specificity"""
        logger.info("Optimizing prediction threshold...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model with class balancing
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight={0: 1, 1: 2},  # Give more weight to inhibitors
            random_state=42
        )
        
        # Cross-validation training
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_scaled, y_train)):
            X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            self.model.fit(X_fold_train, y_fold_train)
            
            # Get prediction probabilities
            y_pred_proba = self.model.predict_proba(X_fold_val)[:, 1]
            
            # Find optimal threshold for this fold
            fold_threshold = self._find_optimal_threshold(y_fold_val, y_pred_proba)
            cv_scores.append(fold_threshold)
        
        # Use mean threshold from CV
        self.optimal_threshold = np.mean(cv_scores)
        
        # Train final model on all data
        self.model.fit(X_train_scaled, y_train)
        
        # Test on training data
        train_proba = self.model.predict_proba(X_train_scaled)[:, 1]
        train_pred = (train_proba >= self.optimal_threshold).astype(int)
        
        # Test on specificity test set
        test_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        test_pred = (test_proba >= self.optimal_threshold).astype(int)
        
        results = {
            'optimal_threshold': self.optimal_threshold,
            'train_metrics': self._calculate_metrics(y_train, train_pred, train_proba),
            'specificity_test_metrics': self._calculate_metrics(y_test, test_pred, test_proba),
            'feature_importance': dict(zip(X_train.columns if hasattr(X_train, 'columns') else range(X_train.shape[1]), 
                                         self.model.feature_importances_))
        }
        
        logger.info(f"Optimal threshold: {self.optimal_threshold:.3f}")
        logger.info(f"Training accuracy: {results['train_metrics']['accuracy']:.3f}")
        logger.info(f"Specificity test accuracy: {results['specificity_test_metrics']['accuracy']:.3f}")
        logger.info(f"Specificity test precision: {results['specificity_test_metrics']['precision']:.3f}")
        
        return results
    
    def _find_optimal_threshold(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Find optimal threshold based on specificity target"""
        thresholds = np.arange(0.1, 0.95, 0.05)
        best_threshold = 0.5
        best_score = 0.0
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            # Calculate metrics
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                # Score based on specificity target and overall balance
                if specificity >= self.specificity_target:
                    score = specificity + 0.5 * sensitivity  # Prioritize specificity
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
        
        return best_threshold
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict:
        """Calculate comprehensive metrics"""
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': cm.tolist()
        }
        
        # Add specificity and sensitivity
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return metrics
    
    def create_threshold_analysis_plots(self, X_train: np.ndarray, y_train: np.ndarray,
                                       X_test: np.ndarray, y_test: np.ndarray) -> None:
        """Create plots showing threshold optimization results"""
        logger.info("Creating threshold analysis plots...")
        
        import os
        os.makedirs('analysis/threshold_plots', exist_ok=True)
        
        # Scale data
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get probabilities
        train_proba = self.model.predict_proba(X_train_scaled)[:, 1]
        test_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Threshold vs Metrics plot
        ax = axes[0, 0]
        thresholds = np.arange(0.1, 0.95, 0.05)
        accuracies, precisions, recalls, specificities = [], [], [], []
        
        for threshold in thresholds:
            y_pred = (train_proba >= threshold).astype(int)
            accuracies.append(accuracy_score(y_train, y_pred))
            precisions.append(precision_score(y_train, y_pred, zero_division=0))
            recalls.append(recall_score(y_train, y_pred, zero_division=0))
            
            cm = confusion_matrix(y_train, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
            else:
                specificities.append(0)
        
        ax.plot(thresholds, accuracies, 'b-', label='Accuracy', alpha=0.8)
        ax.plot(thresholds, precisions, 'r-', label='Precision', alpha=0.8)
        ax.plot(thresholds, recalls, 'g-', label='Recall', alpha=0.8)
        ax.plot(thresholds, specificities, 'm-', label='Specificity', alpha=0.8)
        ax.axvline(x=self.optimal_threshold, color='k', linestyle='--', label=f'Optimal ({self.optimal_threshold:.3f})')
        ax.axhline(y=self.specificity_target, color='orange', linestyle=':', label=f'Target Specificity ({self.specificity_target})')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Threshold vs Metrics (Training Data)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Probability distributions
        ax = axes[0, 1]
        inhibitor_proba = train_proba[y_train == 1]
        non_inhibitor_proba = train_proba[y_train == 0]
        
        ax.hist(non_inhibitor_proba, bins=20, alpha=0.6, label='Non-inhibitors', color='red', density=True)
        ax.hist(inhibitor_proba, bins=20, alpha=0.6, label='Inhibitors', color='blue', density=True)
        ax.axvline(x=self.optimal_threshold, color='k', linestyle='--', label=f'Optimal Threshold')
        ax.set_xlabel('Prediction Probability')
        ax.set_ylabel('Density')
        ax.set_title('Probability Distribution (Training Data)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Specificity test results
        ax = axes[1, 0]
        test_pred = (test_proba >= self.optimal_threshold).astype(int)
        
        # All test samples should be non-inhibitors
        false_positive_count = sum(test_pred)
        true_negative_count = len(test_pred) - false_positive_count
        
        ax.bar(['True Negatives\\n(Correct)', 'False Positives\\n(Incorrect)'], 
               [true_negative_count, false_positive_count],
               color=['green', 'red'], alpha=0.7)
        ax.set_ylabel('Number of Compounds')
        ax.set_title('Specificity Test Results\\n(All compounds should be non-inhibitors)')
        
        # Add percentage labels
        total = len(test_pred)
        ax.text(0, true_negative_count + 0.5, f'{true_negative_count/total:.1%}', ha='center', va='bottom')
        if false_positive_count > 0:
            ax.text(1, false_positive_count + 0.5, f'{false_positive_count/total:.1%}', ha='center', va='bottom')
        
        # 4. Feature importance
        ax = axes[1, 1]
        feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
        if hasattr(X_train, 'columns'):
            feature_names = list(X_train.columns)
        
        importance = self.model.feature_importances_
        sorted_idx = np.argsort(importance)[::-1]
        
        ax.barh(range(len(importance)), importance[sorted_idx])
        ax.set_yticks(range(len(importance)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx])
        ax.set_xlabel('Feature Importance')
        ax.set_title('Feature Importance (Optimized Model)')
        
        plt.tight_layout()
        plt.savefig('analysis/threshold_plots/threshold_optimization_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Threshold analysis plots saved to analysis/threshold_plots/")

def main():
    """Run threshold optimization"""
    optimizer = MDM2ThresholdOptimizer(specificity_target=0.90)
    
    # Load data with problematic features identified by SHAP
    problematic_features = ['hbd', 'num_ro5_violations', 'rtb']
    
    X_train, y_train, X_test, y_test, features = optimizer.load_data(
        'data/processed/balanced_mdm2_data.csv',
        'data/processed/specificity_test_set.csv',
        problematic_features
    )
    
    # Optimize threshold
    results = optimizer.optimize_threshold(X_train, y_train, X_test, y_test)
    
    # Create analysis plots
    optimizer.create_threshold_analysis_plots(X_train, y_train, X_test, y_test)
    
    # Print results
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION RESULTS")
    print("="*80)
    
    print(f"\nOptimal Threshold: {results['optimal_threshold']:.3f}")
    print(f"Features Used: {features}")
    
    print(f"\nTraining Performance:")
    train_metrics = results['train_metrics']
    print(f"  Accuracy:    {train_metrics['accuracy']:.3f}")
    print(f"  Precision:   {train_metrics['precision']:.3f}")
    print(f"  Recall:      {train_metrics['recall']:.3f}")
    print(f"  Specificity: {train_metrics.get('specificity', 'N/A'):.3f}")
    print(f"  F1-Score:    {train_metrics['f1_score']:.3f}")
    
    print(f"\nSpecificity Test Performance:")
    test_metrics = results['specificity_test_metrics']
    print(f"  Accuracy:    {test_metrics['accuracy']:.3f}")
    print(f"  Precision:   {test_metrics['precision']:.3f}")
    print(f"  Specificity: {test_metrics.get('specificity', 'N/A'):.3f}")
    print(f"  False Positive Rate: {test_metrics.get('false_positive_rate', 'N/A'):.3f}")
    
    # Count false positives
    cm = test_metrics['confusion_matrix']
    if len(cm) == 2 and len(cm[0]) == 2:
        tn, fp = cm[0]
        print(f"  True Negatives: {tn}")
        print(f"  False Positives: {fp}")
        print(f"  Specificity Success: {tn}/{tn+fp} compounds correctly identified as non-inhibitors")
    
    print(f"\nFeature Importance:")
    for feature, importance in results['feature_importance'].items():
        print(f"  {feature}: {importance:.4f}")
    
    print(f"\n📊 Analysis plots saved to analysis/threshold_plots/threshold_optimization_analysis.png")

if __name__ == "__main__":
    main()