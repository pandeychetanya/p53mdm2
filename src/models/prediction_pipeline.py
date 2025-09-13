"""
SMILES-to-Prediction Pipeline
Complete pipeline for MDM2 inhibition prediction from SMILES strings
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Union
import logging
import joblib
import pickle
from pathlib import Path

# Import custom modules
from molecular_gnn import MolecularGraphBuilder
from hybrid_model import HybridMDM2Model, create_hybrid_model
import sys
sys.path.append('../data')
sys.path.append('../features')
from data_processor import MDM2DataProcessor
from evolutionary_selector import EvolutionaryFeatureSelector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MDM2InhibitionPredictor:
    """
    Complete prediction pipeline for MDM2 inhibition from SMILES
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the prediction pipeline
        
        Args:
            model_path: Path to saved model weights
        """
        self.model = None
        self.graph_builder = MolecularGraphBuilder()
        self.data_processor = MDM2DataProcessor()
        self.feature_selector = None
        self.scaler = None
        self.selected_features = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def setup_pipeline(self, 
                      dataset_path: str,
                      feature_selection: bool = True,
                      save_components: bool = True):
        """
        Setup the complete pipeline using training data
        
        Args:
            dataset_path: Path to training dataset
            feature_selection: Whether to perform evolutionary feature selection
            save_components: Whether to save pipeline components
        """
        logger.info("Setting up MDM2 prediction pipeline...")
        
        # Load and prepare data
        df = self.data_processor.load_data(dataset_path)
        
        # Feature selection if requested
        if feature_selection:
            logger.info("Performing evolutionary feature selection...")
            
            # Prepare features for selection
            descriptor_columns = [
                'molecular_weight', 'alogp', 'hba', 'hbd', 'psa', 'rtb', 
                'num_ro5_violations', 'qed_weighted', 'cx_most_apka', 
                'cx_most_bpka', 'cx_logp', 'cx_logd', 'aromatic_rings', 
                'heavy_atoms', 'num_alerts'
            ]
            available_columns = [col for col in descriptor_columns if col in df.columns]
            X_features = df[available_columns].fillna(0)
            y = df['is_inhibitor']
            
            # Run feature selection
            self.feature_selector = EvolutionaryFeatureSelector(
                max_features=10,
                population_size=30,
                generations=20
            )
            self.feature_selector.fit(X_features, y)
            
            # Get best features
            best_solutions = self.feature_selector.get_best_features(n_solutions=1)
            self.selected_features = best_solutions[0]['features']
            logger.info(f"Selected features: {self.selected_features}")
        else:
            # Use all available descriptors
            self.selected_features = [
                'alogp', 'hba', 'hbd', 'rtb', 'num_ro5_violations', 
                'cx_logp', 'aromatic_rings'
            ]
        
        # Prepare dataset with selected features
        X_train, X_test, y_train, y_test, _, _ = self.data_processor.prepare_dataset(
            dataset_path, 
            selected_features=self.selected_features,
            use_rdkit=False
        )
        
        # Create model
        self.model = create_hybrid_model(descriptor_dim=len(self.selected_features))
        self.model.to(self.device)
        
        # Store scaler
        self.scaler = self.data_processor.scaler
        
        logger.info(f"Pipeline setup complete. Model on device: {self.device}")
        
        # Save components if requested
        if save_components:
            self.save_pipeline_components()
        
        return X_train, X_test, y_train, y_test
    
    def predict_smiles(self, 
                      smiles: Union[str, List[str]], 
                      return_confidence: bool = True) -> Dict:
        """
        Predict MDM2 inhibition for SMILES string(s)
        
        Args:
            smiles: SMILES string or list of SMILES strings
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary with predictions and optional confidence scores
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please setup pipeline or load model first.")
        
        # Ensure input is list
        if isinstance(smiles, str):
            smiles = [smiles]
        
        results = {
            'smiles': smiles,
            'predictions': [],
            'binary_predictions': [],
            'confidence_scores': [] if return_confidence else None
        }
        
        for smi in smiles:
            try:
                # Process single SMILES
                prediction, confidence = self._predict_single_smiles(smi, return_confidence)
                
                results['predictions'].append(prediction)
                results['binary_predictions'].append('Inhibitor' if prediction > 0.5 else 'Non-inhibitor')
                
                if return_confidence:
                    results['confidence_scores'].append(confidence)
                
            except Exception as e:
                logger.error(f"Error predicting SMILES {smi}: {e}")
                results['predictions'].append(None)
                results['binary_predictions'].append('Error')
                if return_confidence:
                    results['confidence_scores'].append(None)
        
        return results
    
    def _predict_single_smiles(self, smiles: str, return_confidence: bool = True) -> Tuple[float, Optional[float]]:
        """
        Predict for a single SMILES string
        
        Args:
            smiles: SMILES string
            return_confidence: Whether to return confidence
            
        Returns:
            Tuple of (prediction, confidence)
        """
        self.model.eval()
        
        with torch.no_grad():
            # Convert SMILES to graph
            graph_data = self.graph_builder.smiles_to_graph(smiles)
            if graph_data is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            
            # Add batch dimension
            graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long)
            graph_data = graph_data.to(self.device)
            
            # Process descriptors
            descriptor_features = self.data_processor.process_new_smiles(smiles, use_rdkit=False)
            descriptor_tensor = torch.FloatTensor(descriptor_features).unsqueeze(0).to(self.device)
            
            # Predict
            if return_confidence:
                prediction, confidence = self.model.predict_with_confidence(graph_data, descriptor_tensor)
                return prediction.item(), confidence.item()
            else:
                prediction = self.model.forward(graph_data, descriptor_tensor)
                return prediction.item(), None
    
    def batch_predict(self, 
                     smiles_list: List[str], 
                     batch_size: int = 32,
                     return_confidence: bool = True) -> Dict:
        """
        Batch prediction for efficiency with large lists
        
        Args:
            smiles_list: List of SMILES strings
            batch_size: Batch size for processing
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary with batch predictions
        """
        results = {
            'smiles': [],
            'predictions': [],
            'binary_predictions': [],
            'confidence_scores': [] if return_confidence else None
        }
        
        # Process in batches
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i + batch_size]
            batch_results = self.predict_smiles(batch_smiles, return_confidence)
            
            results['smiles'].extend(batch_results['smiles'])
            results['predictions'].extend(batch_results['predictions'])
            results['binary_predictions'].extend(batch_results['binary_predictions'])
            
            if return_confidence:
                results['confidence_scores'].extend(batch_results['confidence_scores'])
        
        return results
    
    def save_model(self, model_path: str):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'selected_features': self.selected_features,
            'model_config': {
                'descriptor_dim': len(self.selected_features) if self.selected_features else 7
            }
        }, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Restore selected features
        self.selected_features = checkpoint.get('selected_features', 
                                               ['alogp', 'hba', 'hbd', 'rtb', 'num_ro5_violations', 'cx_logp', 'aromatic_rings'])
        
        # Create model
        model_config = checkpoint.get('model_config', {'descriptor_dim': 7})
        self.model = create_hybrid_model(**model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded from {model_path}")
    
    def save_pipeline_components(self, base_path: str = "../../models/"):
        """Save pipeline components for reproducibility"""
        Path(base_path).mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        if self.scaler:
            joblib.dump(self.scaler, f"{base_path}/scaler.pkl")
        
        # Save feature selector
        if self.feature_selector:
            with open(f"{base_path}/feature_selector.pkl", 'wb') as f:
                pickle.dump(self.feature_selector, f)
        
        # Save selected features
        if self.selected_features:
            with open(f"{base_path}/selected_features.pkl", 'wb') as f:
                pickle.dump(self.selected_features, f)
        
        logger.info(f"Pipeline components saved to {base_path}")
    
    def load_pipeline_components(self, base_path: str = "../../models/"):
        """Load pipeline components"""
        # Load scaler
        scaler_path = f"{base_path}/scaler.pkl"
        if Path(scaler_path).exists():
            self.scaler = joblib.load(scaler_path)
            self.data_processor.scaler = self.scaler
        
        # Load feature selector
        selector_path = f"{base_path}/feature_selector.pkl"
        if Path(selector_path).exists():
            with open(selector_path, 'rb') as f:
                self.feature_selector = pickle.load(f)
        
        # Load selected features
        features_path = f"{base_path}/selected_features.pkl"
        if Path(features_path).exists():
            with open(features_path, 'rb') as f:
                self.selected_features = pickle.load(f)
            self.data_processor.selected_features = self.selected_features
        
        logger.info(f"Pipeline components loaded from {base_path}")

def format_prediction_output(results: Dict, detailed: bool = False) -> str:
    """
    Format prediction results for user-friendly output
    
    Args:
        results: Results dictionary from prediction
        detailed: Whether to show detailed information
        
    Returns:
        Formatted string output
    """
    output_lines = []
    output_lines.append("=" * 60)
    output_lines.append("MDM2 INHIBITION PREDICTION RESULTS")
    output_lines.append("=" * 60)
    
    for i, smiles in enumerate(results['smiles']):
        if results['predictions'][i] is None:
            output_lines.append(f"\nSMILES: {smiles}")
            output_lines.append("Result: ERROR - Invalid SMILES")
            continue
        
        prediction = results['predictions'][i]
        binary_result = results['binary_predictions'][i]
        confidence = results['confidence_scores'][i] if results['confidence_scores'] else None
        
        output_lines.append(f"\nSMILES: {smiles}")
        output_lines.append(f"Prediction: {binary_result}")
        output_lines.append(f"Probability: {prediction:.3f}")
        
        if confidence is not None:
            output_lines.append(f"Confidence: {confidence:.3f}")
            
        if detailed:
            output_lines.append(f"Raw Score: {prediction:.6f}")
    
    output_lines.append("\n" + "=" * 60)
    return "\n".join(output_lines)

def main():
    """Test the prediction pipeline"""
    # Create predictor
    predictor = MDM2InhibitionPredictor()
    
    # Test SMILES strings
    test_smiles = [
        "CCc1ccc2nc(N3CCN(C(=O)c4ccc(F)cc4)CC3)nc2c1",  # Example inhibitor-like
        "CCO",  # Simple molecule (ethanol)
        "c1ccccc1"  # Benzene
    ]
    
    print("Testing MDM2 Inhibition Prediction Pipeline")
    print("=" * 50)
    
    try:
        # Setup pipeline (this would normally use real training data)
        logger.info("Setting up mock pipeline for testing...")
        predictor.selected_features = ['alogp', 'hba', 'hbd', 'rtb', 'num_ro5_violations', 'cx_logp', 'aromatic_rings']
        predictor.model = create_hybrid_model(descriptor_dim=7)
        predictor.model.eval()
        
        # Mock scaler setup
        from sklearn.preprocessing import StandardScaler
        predictor.scaler = StandardScaler()
        predictor.scaler.mean_ = np.zeros(7)
        predictor.scaler.scale_ = np.ones(7)
        predictor.data_processor.scaler = predictor.scaler
        predictor.data_processor.selected_features = predictor.selected_features
        
        # Make predictions
        results = predictor.predict_smiles(test_smiles, return_confidence=True)
        
        # Format and display results
        formatted_output = format_prediction_output(results, detailed=True)
        print(formatted_output)
        
    except Exception as e:
        logger.error(f"Error in pipeline test: {e}")
        print(f"Pipeline test failed: {e}")

if __name__ == "__main__":
    main()