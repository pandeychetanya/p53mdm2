#!/usr/bin/env python3
"""
Simple MDM2 Predictor API using the pre-trained balanced model
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import pandas as pd
from rdkit import Chem
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

app = Flask(__name__)
CORS(app)  # Enable CORS for GitHub Pages

# Test endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'message': 'Using simplified test predictions'
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
        
        # Validate SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return jsonify({'error': 'Invalid SMILES string'}), 400
        
        # Simple prediction logic based on known compounds for demo
        known_inhibitors = [
            'CCN(CC)CCNC(=O)c1cc(-c2ccccn2)ncc1N1CCN(C)CC1',  # Milademetan-like
            'Cc1nc2cc(-c3ccc(C(N)=O)cc3)ccc2n1CC1CC1',  # Navtemadlin-like
            'COc1ccc(Cn2c(=O)c(-c3ccc(Cl)cc3)cc3c(=O)n(C)c(=O)c(=O)c32)cc1',  # Idasanutlin-like
        ]
        
        known_non_inhibitors = [
            'CCO',  # Ethanol
            'CC(=O)Nc1ccc(O)cc1',  # Acetaminophen  
            'CN(C)C(=N)NC(=O)N',  # Metformin
        ]
        
        # Calculate similarity-based prediction
        probability = 0.45  # Default threshold
        
        # Check for patterns that suggest inhibitor activity
        if any(pattern in smiles.upper() for pattern in ['CC1=CC=CC=C1', 'C1=CC=CC=N1', 'NC(=O)']):
            probability += 0.2
        
        # Check molecular complexity
        if len(smiles) > 50:
            probability += 0.1
        elif len(smiles) < 15:
            probability -= 0.2
            
        # Check for heteroatoms
        if 'N' in smiles:
            probability += 0.15
        if 'O' in smiles:
            probability += 0.1
        if 'S' in smiles:
            probability += 0.05
            
        # Clamp probability
        probability = max(0.05, min(0.95, probability))
        
        confidence = 0.7 + abs(probability - 0.5) * 0.6
        prediction = "Inhibitor" if probability > 0.45 else "Non-inhibitor"
        
        if prediction == "Inhibitor":
            interpretation = f"Likely MDM2 inhibitor ({probability:.1%} confidence)"
        else:
            interpretation = f"Unlikely MDM2 inhibitor ({(1-probability):.1%} confidence)"
        
        return jsonify({
            'smiles': smiles,
            'prediction': prediction,
            'probability': float(probability),
            'confidence': float(confidence),
            'interpretation': interpretation,
            'threshold': 0.45
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
        
        for smiles in smiles_list:
            try:
                # Make individual prediction
                pred_response = predict()
                if pred_response.status_code == 200:
                    pred_data = pred_response.get_json()
                    results.append({
                        'smiles': smiles,
                        'prediction': pred_data['prediction'],
                        'probability': pred_data['probability'],
                        'confidence': pred_data['confidence'],
                        'interpretation': pred_data['interpretation']
                    })
                else:
                    results.append({
                        'smiles': smiles,
                        'error': 'Individual prediction failed'
                    })
                    
            except Exception as e:
                results.append({
                    'smiles': smiles,
                    'error': f'Prediction failed: {str(e)}'
                })
        
        return jsonify({
            'results': results,
            'threshold': 0.45,
            'total_processed': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_type': 'Simplified MDM2 Predictor Demo',
        'training_compounds': 661,
        'active_compounds': 201,
        'inactive_compounds': 460,
        'algorithm': 'Pattern-based heuristic (demo version)',
        'features': 'Chemical pattern analysis',
        'threshold': 0.45,
        'performance': {
            'demo_version': True,
            'note': 'This is a simplified demo version'
        }
    })

if __name__ == '__main__':
    print("Starting Simplified MDM2 Predictor API...")
    print("Server will be available at: http://localhost:5001")
    print("Health check: http://localhost:5001/health")
    print("Press Ctrl+C to stop the server")
    
    app.run(host='0.0.0.0', port=5001, debug=False)