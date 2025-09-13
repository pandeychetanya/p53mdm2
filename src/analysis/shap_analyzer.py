"""
SHAP Feature Importance Analyzer for MDM2 Inhibitor Prediction
Analyzes feature importance patterns for inhibitors vs non-inhibitors to identify overfitting
"""

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SHAPFeatureAnalyzer:
    """Analyzes feature importance using SHAP values to identify problematic features"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.rf_model = None
        self.lr_model = None
        self.shap_explainer_rf = None
        self.shap_explainer_lr = None
        self.feature_names = None
        
    def load_and_prepare_data(self, data_path: str, selected_features: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load and prepare data for SHAP analysis"""
        logger.info(f"Loading data from {data_path}")
        
        df = pd.read_csv(data_path)
        
        # Use selected features from evolutionary algorithm
        if selected_features is None:
            selected_features = ['alogp', 'hba', 'hbd', 'rtb', 'num_ro5_violations', 'cx_logp', 'aromatic_rings']
        
        # Filter available features
        available_features = [f for f in selected_features if f in df.columns]
        logger.info(f"Using {len(available_features)} features: {available_features}")
        
        # Prepare feature matrix
        X = df[available_features].fillna(0)
        y = df['is_inhibitor'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        self.feature_names = available_features
        
        logger.info(f"Dataset shape: {X_scaled.shape}")
        logger.info(f"Class distribution: {np.bincount(y)}")
        
        return X_scaled, y, available_features
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train models for SHAP analysis"""
        logger.info("Training models for SHAP analysis...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42,
            class_weight='balanced'
        )
        self.rf_model.fit(X_train, y_train)
        
        # Train Logistic Regression
        self.lr_model = LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        )
        self.lr_model.fit(X_train, y_train)
        
        # Create SHAP explainers
        self.shap_explainer_rf = shap.TreeExplainer(self.rf_model)
        self.shap_explainer_lr = shap.LinearExplainer(self.lr_model, X_train)
        
        # Store test data for analysis
        self.X_test = X_test
        self.y_test = y_test
        
        logger.info("Models trained successfully")
    
    def analyze_feature_importance(self, save_plots: bool = True) -> Dict:
        """Analyze feature importance using SHAP values"""
        logger.info("Computing SHAP values...")
        
        # Compute SHAP values
        shap_values_rf = self.shap_explainer_rf.shap_values(self.X_test)
        shap_values_lr = self.shap_explainer_lr.shap_values(self.X_test)
        
        # Debug initial shapes
        logger.info(f"Initial RF SHAP shape: {np.array(shap_values_rf).shape}")
        logger.info(f"Initial LR SHAP shape: {np.array(shap_values_lr).shape}")
        
        # For RF, handle different SHAP output formats
        shap_values_rf = np.array(shap_values_rf)
        if len(shap_values_rf.shape) == 3:  # (n_samples, n_features, n_classes)
            # Take class 1 (inhibitor) SHAP values
            shap_values_rf = shap_values_rf[:, :, 1]
        elif isinstance(shap_values_rf, list) and len(shap_values_rf) == 2:
            shap_values_rf = shap_values_rf[1]
        
        # Ensure arrays are 2D
        shap_values_rf = np.array(shap_values_rf)
        shap_values_lr = np.array(shap_values_lr)
        
        # Final shapes
        logger.info(f"Final RF SHAP shape: {shap_values_rf.shape}")
        logger.info(f"Final LR SHAP shape: {shap_values_lr.shape}")
        
        results = {
            'rf_feature_importance': self._analyze_shap_patterns(shap_values_rf, 'Random Forest'),
            'lr_feature_importance': self._analyze_shap_patterns(shap_values_lr, 'Logistic Regression'),
            'problematic_features': self._identify_problematic_features(shap_values_rf, shap_values_lr)
        }
        
        if save_plots:
            self._create_visualizations(shap_values_rf, shap_values_lr)
        
        return results
    
    def _analyze_shap_patterns(self, shap_values: np.ndarray, model_name: str) -> Dict:
        """Analyze SHAP value patterns for inhibitors vs non-inhibitors"""
        
        # Separate by class
        inhibitor_mask = self.y_test == 1
        non_inhibitor_mask = self.y_test == 0
        
        inhibitor_shap = shap_values[inhibitor_mask]
        non_inhibitor_shap = shap_values[non_inhibitor_mask]
        
        analysis = {
            'model': model_name,
            'feature_importance_global': np.mean(np.abs(shap_values), axis=0),
            'inhibitor_patterns': {
                'mean_shap': np.mean(inhibitor_shap, axis=0),
                'std_shap': np.std(inhibitor_shap, axis=0),
                'positive_contributions': np.mean(inhibitor_shap > 0, axis=0)
            },
            'non_inhibitor_patterns': {
                'mean_shap': np.mean(non_inhibitor_shap, axis=0),
                'std_shap': np.std(non_inhibitor_shap, axis=0),
                'positive_contributions': np.mean(non_inhibitor_shap > 0, axis=0)
            }
        }
        
        # Create feature ranking
        feature_ranking = []
        for i, feature in enumerate(self.feature_names):
            feature_ranking.append({
                'feature': feature,
                'global_importance': float(analysis['feature_importance_global'][i]),
                'inhibitor_mean': float(analysis['inhibitor_patterns']['mean_shap'][i]),
                'non_inhibitor_mean': float(analysis['non_inhibitor_patterns']['mean_shap'][i]),
                'contribution_difference': float(
                    analysis['inhibitor_patterns']['mean_shap'][i] - 
                    analysis['non_inhibitor_patterns']['mean_shap'][i]
                )
            })
        
        # Sort by global importance
        feature_ranking.sort(key=lambda x: x['global_importance'], reverse=True)
        analysis['feature_ranking'] = feature_ranking
        
        return analysis
    
    def _identify_problematic_features(self, shap_values_rf: np.ndarray, shap_values_lr: np.ndarray) -> Dict:
        """Identify features that might cause overfitting or false positives"""
        
        inhibitor_mask = self.y_test == 1
        non_inhibitor_mask = self.y_test == 0
        
        problematic_features = []
        
        for i, feature in enumerate(self.feature_names):
            # RF analysis
            rf_inhibitor_mean = np.mean(shap_values_rf[inhibitor_mask, i])
            rf_non_inhibitor_mean = np.mean(shap_values_rf[non_inhibitor_mask, i])
            rf_difference = rf_inhibitor_mean - rf_non_inhibitor_mean
            
            # LR analysis
            lr_inhibitor_mean = np.mean(shap_values_lr[inhibitor_mask, i])
            lr_non_inhibitor_mean = np.mean(shap_values_lr[non_inhibitor_mask, i])
            lr_difference = lr_inhibitor_mean - lr_non_inhibitor_mean
            
            # Check for potential issues
            issues = []
            
            # Feature has positive contribution for non-inhibitors (false positive risk)
            if rf_non_inhibitor_mean > 0.01:
                issues.append("Positive contribution for non-inhibitors (RF)")
            if lr_non_inhibitor_mean > 0.01:
                issues.append("Positive contribution for non-inhibitors (LR)")
            
            # Feature shows inconsistent patterns between models
            if (rf_difference > 0) != (lr_difference > 0) and abs(rf_difference) > 0.01:
                issues.append("Inconsistent between RF and LR models")
            
            # High variance in contributions (unstable feature)
            rf_variance = np.var(shap_values_rf[:, i])
            if rf_variance > 0.1:
                issues.append("High variance in SHAP values")
            
            if issues:
                problematic_features.append({
                    'feature': feature,
                    'issues': issues,
                    'rf_inhibitor_mean': rf_inhibitor_mean,
                    'rf_non_inhibitor_mean': rf_non_inhibitor_mean,
                    'lr_inhibitor_mean': lr_inhibitor_mean,
                    'lr_non_inhibitor_mean': lr_non_inhibitor_mean,
                    'rf_variance': rf_variance
                })
        
        return {
            'problematic_features': problematic_features,
            'recommendations': self._generate_recommendations(problematic_features)
        }
    
    def _generate_recommendations(self, problematic_features: List[Dict]) -> List[str]:
        """Generate recommendations based on problematic features"""
        recommendations = []
        
        if not problematic_features:
            recommendations.append("No major problematic features identified. Model features appear well-calibrated.")
            return recommendations
        
        # Check for features with false positive risk
        fp_risk_features = [f for f in problematic_features 
                           if any("Positive contribution for non-inhibitors" in issue for issue in f['issues'])]
        
        if fp_risk_features:
            feature_names = [f['feature'] for f in fp_risk_features]
            recommendations.append(f"Remove or re-weight features that contribute positively to non-inhibitors: {feature_names}")
        
        # Check for unstable features
        unstable_features = [f for f in problematic_features 
                            if any("High variance" in issue for issue in f['issues'])]
        
        if unstable_features:
            feature_names = [f['feature'] for f in unstable_features]
            recommendations.append(f"Consider feature transformation or removal for high-variance features: {feature_names}")
        
        # Check for model inconsistencies
        inconsistent_features = [f for f in problematic_features 
                               if any("Inconsistent between" in issue for issue in f['issues'])]
        
        if inconsistent_features:
            feature_names = [f['feature'] for f in inconsistent_features]
            recommendations.append(f"Investigate model-inconsistent features for potential data quality issues: {feature_names}")
        
        return recommendations
    
    def _create_visualizations(self, shap_values_rf: np.ndarray, shap_values_lr: np.ndarray) -> None:
        """Create SHAP visualization plots"""
        logger.info("Creating SHAP visualizations...")
        
        # Create output directory
        import os
        os.makedirs('analysis/shap_plots', exist_ok=True)
        
        # Global feature importance plot
        plt.figure(figsize=(12, 8))
        
        # RF importance
        plt.subplot(2, 2, 1)
        rf_importance = np.mean(np.abs(shap_values_rf), axis=0)
        plt.barh(self.feature_names, rf_importance)
        plt.title('Random Forest - Global Feature Importance')
        plt.xlabel('Mean |SHAP Value|')
        
        # LR importance
        plt.subplot(2, 2, 2)
        lr_importance = np.mean(np.abs(shap_values_lr), axis=0)
        plt.barh(self.feature_names, lr_importance)
        plt.title('Logistic Regression - Global Feature Importance')
        plt.xlabel('Mean |SHAP Value|')
        
        # Comparison by class
        plt.subplot(2, 2, 3)
        inhibitor_mask = self.y_test == 1
        non_inhibitor_mask = self.y_test == 0
        
        inhibitor_mean = np.mean(shap_values_rf[inhibitor_mask], axis=0)
        non_inhibitor_mean = np.mean(shap_values_rf[non_inhibitor_mask], axis=0)
        
        x = np.arange(len(self.feature_names))
        width = 0.35
        
        plt.bar(x - width/2, inhibitor_mean, width, label='Inhibitors', alpha=0.8)
        plt.bar(x + width/2, non_inhibitor_mean, width, label='Non-inhibitors', alpha=0.8)
        plt.xlabel('Features')
        plt.ylabel('Mean SHAP Value')
        plt.title('RF - Mean SHAP Values by Class')
        plt.xticks(x, self.feature_names, rotation=45)
        plt.legend()
        
        # Feature correlation with predictions
        plt.subplot(2, 2, 4)
        feature_means = np.mean(self.X_test, axis=0)
        shap_means = np.mean(shap_values_rf, axis=0)
        
        plt.scatter(feature_means, shap_means, alpha=0.7)
        for i, feature in enumerate(self.feature_names):
            plt.annotate(feature, (feature_means[i], shap_means[i]), fontsize=8)
        plt.xlabel('Feature Mean Value')
        plt.ylabel('Mean SHAP Value')
        plt.title('Feature Values vs SHAP Contributions')
        
        plt.tight_layout()
        plt.savefig('analysis/shap_plots/feature_importance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("SHAP visualizations saved to analysis/shap_plots/")

def main():
    """Run SHAP analysis on MDM2 dataset"""
    analyzer = SHAPFeatureAnalyzer()
    
    # Load balanced dataset
    X, y, features = analyzer.load_and_prepare_data("data/processed/balanced_mdm2_data.csv")
    
    # Train models
    analyzer.train_models(X, y)
    
    # Analyze feature importance
    results = analyzer.analyze_feature_importance(save_plots=True)
    
    # Print results
    print("\n" + "="*80)
    print("SHAP FEATURE IMPORTANCE ANALYSIS RESULTS")
    print("="*80)
    
    # Random Forest Analysis
    rf_results = results['rf_feature_importance']
    print(f"\nRANDOM FOREST FEATURE RANKING:")
    print("-" * 50)
    for i, feature_info in enumerate(rf_results['feature_ranking'][:10]):
        print(f"{i+1:2d}. {feature_info['feature']:20} | "
              f"Importance: {feature_info['global_importance']:.4f} | "
              f"Inhibitor: {feature_info['inhibitor_mean']:+.4f} | "
              f"Non-inhibitor: {feature_info['non_inhibitor_mean']:+.4f}")
    
    # Problematic Features
    problems = results['problematic_features']
    print(f"\nPROBLEMATIC FEATURES ANALYSIS:")
    print("-" * 50)
    
    if problems['problematic_features']:
        for feature_info in problems['problematic_features']:
            print(f"\n❌ {feature_info['feature']}:")
            for issue in feature_info['issues']:
                print(f"   - {issue}")
            print(f"   RF: Inhibitor {feature_info['rf_inhibitor_mean']:+.4f}, "
                  f"Non-inhibitor {feature_info['rf_non_inhibitor_mean']:+.4f}")
    else:
        print("✅ No major problematic features identified!")
    
    print(f"\nRECOMMENDATIONS:")
    print("-" * 50)
    for i, rec in enumerate(problems['recommendations'], 1):
        print(f"{i}. {rec}")
    
    print(f"\n📊 Visualizations saved to analysis/shap_plots/feature_importance_analysis.png")

if __name__ == "__main__":
    main()