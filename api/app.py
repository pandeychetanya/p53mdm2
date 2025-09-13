from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from features.enhanced_descriptors import EnhancedMolecularDescriptors
from models.balanced_mdm2_predictor import BalancedMDM2Predictor

app = Flask(__name__)
CORS(app)  # Enable CORS for GitHub Pages

# Global variables for model and descriptor generator
model = None
descriptor_gen = None

def load_model():
    """Load the trained model and descriptor generator"""
    global model, descriptor_gen
    
    try:
        # Load the balanced predictor
        model = BalancedMDM2Predictor()
        
        # Load pre-trained model if it exists
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved', 'balanced_mdm2_model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                model.ensemble = model_data['ensemble']
                model.feature_selector = model_data['feature_selector']
                model.scaler = model_data['scaler']
                model.threshold = model_data.get('threshold', 0.45)
                model.is_trained = True
        else:
            # Train the model if not pre-trained
            print("Training model...")
            dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'balanced_mdm2_data.csv')
            if os.path.exists(dataset_path):
                data = pd.read_csv(dataset_path)
                # Drop non-feature columns to get only descriptors
                feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                # Remove target and identifier columns
                exclude_cols = ['is_inhibitor', 'pActivity', 'standard_value', 'value', 'pchembl_value']
                feature_cols = [col for col in feature_cols if col not in exclude_cols]
                X = data[feature_cols]
                y = data['is_inhibitor']
                model.train_balanced_ensemble(X, y)
                
                # Save the trained model
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                model_data = {
                    'ensemble': model.ensemble,
                    'feature_selector': model.feature_selector,
                    'scaler': model.scaler,
                    'threshold': model.threshold
                }
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data, f)
            else:
                raise FileNotFoundError("Training data not found")
        
        # Initialize descriptor generator
        descriptor_gen = EnhancedMolecularDescriptors()
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    return True

def smiles_to_features(smiles):
    """Convert SMILES to feature vector"""
    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "Invalid SMILES string"
        
        # Generate features
        features = descriptor_gen.generate_features(smiles)
        if features is None:
            return None, "Failed to generate molecular features"
        
        return features, None
        
    except Exception as e:
        return None, f"Error processing SMILES: {str(e)}"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None and model.is_trained
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        data = request.json
        
        if not data or 'smiles' not in data:
            return jsonify({'error': 'SMILES string required'}), 400
        
        smiles = data['smiles'].strip()
        
        if not smiles:
            return jsonify({'error': 'Empty SMILES string'}), 400
        
        # Convert SMILES to features
        features, error = smiles_to_features(smiles)
        if error:
            return jsonify({'error': error}), 400
        
        # Make prediction
        features_df = pd.DataFrame([features])
        probability = model.predict_proba(features_df)[0]
        prediction = model.predict(features_df)[0]
        
        # Get confidence and interpretation
        confidence = max(probability, 1 - probability)
        
        if prediction == 1:
            result = "Inhibitor"
            interpretation = f"Likely MDM2 inhibitor ({probability:.1%} confidence)"
        else:
            result = "Non-inhibitor" 
            interpretation = f"Unlikely MDM2 inhibitor ({(1-probability):.1%} confidence)"
        
        return jsonify({
            'smiles': smiles,
            'prediction': result,
            'probability': float(probability),
            'confidence': float(confidence),
            'interpretation': interpretation,
            'threshold': model.threshold
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        data = request.json
        
        if not data or 'smiles_list' not in data:
            return jsonify({'error': 'smiles_list required'}), 400
        
        smiles_list = data['smiles_list']
        
        if not isinstance(smiles_list, list) or len(smiles_list) == 0:
            return jsonify({'error': 'smiles_list must be a non-empty list'}), 400
        
        if len(smiles_list) > 100:
            return jsonify({'error': 'Maximum 100 SMILES allowed per batch'}), 400
        
        results = []
        
        for i, smiles in enumerate(smiles_list):
            try:
                smiles = smiles.strip()
                
                # Convert SMILES to features
                features, error = smiles_to_features(smiles)
                if error:
                    results.append({
                        'smiles': smiles,
                        'error': error
                    })
                    continue
                
                # Make prediction
                features_df = pd.DataFrame([features])
                probability = model.predict_proba(features_df)[0]
                prediction = model.predict(features_df)[0]
                
                # Get confidence and interpretation
                confidence = max(probability, 1 - probability)
                
                if prediction == 1:
                    result = "Inhibitor"
                    interpretation = f"Likely MDM2 inhibitor ({probability:.1%} confidence)"
                else:
                    result = "Non-inhibitor"
                    interpretation = f"Unlikely MDM2 inhibitor ({(1-probability):.1%} confidence)"
                
                results.append({
                    'smiles': smiles,
                    'prediction': result,
                    'probability': float(probability),
                    'confidence': float(confidence),
                    'interpretation': interpretation
                })
                
            except Exception as e:
                results.append({
                    'smiles': smiles,
                    'error': f'Prediction failed: {str(e)}'
                })
        
        return jsonify({
            'results': results,
            'threshold': model.threshold,
            'total_processed': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_type': 'Balanced MDM2 Predictor',
        'training_compounds': 661,
        'active_compounds': 201,
        'inactive_compounds': 460,
        'algorithm': 'Ensemble (Balanced Random Forest + Cost-Sensitive Random Forest)',
        'features': '2529 enhanced molecular descriptors',
        'threshold': model.threshold if model else 0.45,
        'performance': {
            'f1_score': 0.662,
            'sensitivity': 0.579,
            'specificity': 0.480,
            'balanced_accuracy': 0.529
        }
    })

if __name__ == '__main__':
    print("Loading MDM2 predictor model...")
    if load_model():
        print("Starting Flask API server...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Failed to load model. Exiting.")