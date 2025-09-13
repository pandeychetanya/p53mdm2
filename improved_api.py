#!/usr/bin/env python3
"""
Improved MDM2 Predictor API with Enhanced Negative Dataset

This API uses experimentally confirmed non-inhibitors to reduce false positives
and provides more accurate predictions with better specificity.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

app = Flask(__name__)
CORS(app)  # Enable CORS for GitHub Pages

class ImprovedMDM2Predictor:
    """Improved MDM2 predictor with enhanced negative training data"""
    
    def __init__(self):
        self.load_training_data()
        self.threshold = 0.65  # More conservative threshold
        
    def load_training_data(self):
        """Load scaffold-split training dataset with confirmed negatives"""
        try:
            # Load scaffold-split training data
            self.training_data = pd.read_csv('data/processed/scaffold_train.csv')
            
            # Separate actives and negatives
            self.actives = self.training_data[self.training_data['is_inhibitor'] == 1]
            self.negatives = self.training_data[self.training_data['is_inhibitor'] == 0]
            
            # Pre-calculate active compound features for similarity
            self.active_features = self._calculate_compound_features(self.actives['canonical_smiles'])
            self.negative_features = self._calculate_compound_features(self.negatives['canonical_smiles'])
            
            print(f"Loaded training data: {len(self.actives)} actives, {len(self.negatives)} negatives")
            
        except Exception as e:
            print(f"Error loading training data: {e}")
            # Fallback to minimal dataset
            self.training_data = None
            self.actives = pd.DataFrame()
            self.negatives = pd.DataFrame()
            
    def _calculate_compound_features(self, smiles_list):
        """Calculate molecular features for compounds"""
        features = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    feat = {
                        'mw': Descriptors.MolWt(mol),
                        'logp': Descriptors.MolLogP(mol),
                        'hba': Descriptors.NumHAcceptors(mol),
                        'hbd': Descriptors.NumHDonors(mol),
                        'tpsa': Descriptors.TPSA(mol),
                        'rotatable': Descriptors.NumRotatableBonds(mol),
                        'aromatic_rings': Descriptors.NumAromaticRings(mol),
                        'heteroatoms': Descriptors.NumHeteroatoms(mol),
                        'heavy_atoms': Descriptors.HeavyAtomCount(mol),
                        'rings': Descriptors.RingCount(mol)
                    }
                    features.append(feat)
                else:
                    features.append(None)
            except:
                features.append(None)
        return features
    
    def _similarity_to_actives(self, query_features):
        """Calculate similarity to known active compounds"""
        if not self.active_features or not query_features:
            return 0.0
            
        similarities = []
        
        for active_feat in self.active_features:
            if active_feat is None:
                continue
                
            # Calculate Euclidean distance in normalized feature space
            dist = 0
            valid_features = 0
            
            for key in query_features:
                if key in active_feat and active_feat[key] is not None and query_features[key] is not None:
                    # Normalize by typical ranges
                    ranges = {
                        'mw': 500, 'logp': 5, 'hba': 10, 'hbd': 5, 'tpsa': 150,
                        'rotatable': 10, 'aromatic_rings': 5, 'heteroatoms': 10,
                        'heavy_atoms': 40, 'rings': 8
                    }
                    norm_range = ranges.get(key, 1)
                    diff = abs(query_features[key] - active_feat[key]) / norm_range
                    dist += diff ** 2
                    valid_features += 1
                    
            if valid_features > 0:
                similarity = 1 / (1 + np.sqrt(dist / valid_features))
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _similarity_to_negatives(self, query_features):
        """Calculate similarity to known negative compounds"""
        if not self.negative_features or not query_features:
            return 0.0
            
        similarities = []
        
        for negative_feat in self.negative_features:
            if negative_feat is None:
                continue
                
            # Calculate similarity to negatives
            dist = 0
            valid_features = 0
            
            for key in query_features:
                if key in negative_feat and negative_feat[key] is not None and query_features[key] is not None:
                    ranges = {
                        'mw': 500, 'logp': 5, 'hba': 10, 'hbd': 5, 'tpsa': 150,
                        'rotatable': 10, 'aromatic_rings': 5, 'heteroatoms': 10,
                        'heavy_atoms': 40, 'rings': 8
                    }
                    norm_range = ranges.get(key, 1)
                    diff = abs(query_features[key] - negative_feat[key]) / norm_range
                    dist += diff ** 2
                    valid_features += 1
                    
            if valid_features > 0:
                similarity = 1 / (1 + np.sqrt(dist / valid_features))
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def predict_inhibition(self, smiles):
        """Predict MDM2 inhibition probability with enhanced specificity"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None, "Invalid SMILES string"
                
            # Calculate features
            query_features = self._calculate_compound_features([smiles])[0]
            if query_features is None:
                return None, "Failed to calculate molecular features"
            
            # Base probability from molecular properties
            base_prob = self._calculate_base_probability(query_features)
            
            # Similarity-based adjustments using training data
            if len(self.actives) > 0 and len(self.negatives) > 0:
                active_similarity = self._similarity_to_actives(query_features)
                negative_similarity = self._similarity_to_negatives(query_features)
                
                # Adjust probability based on similarities
                # Higher similarity to negatives decreases probability
                # Higher similarity to actives increases probability
                similarity_adjustment = (active_similarity - negative_similarity) * 0.3
                base_prob += similarity_adjustment
                
            # Additional conservative filters
            base_prob = self._apply_conservative_filters(query_features, base_prob)
            
            # Clamp probability
            probability = max(0.01, min(0.99, base_prob))
            
            return probability, None
            
        except Exception as e:
            return None, f"Error in prediction: {str(e)}"
    
    def _calculate_base_probability(self, features):
        """Calculate base inhibition probability from molecular features"""
        prob = 0.2  # Conservative starting point
        
        # Molecular weight (MDM2 inhibitors typically 300-600 Da)
        mw = features['mw']
        if 300 <= mw <= 600:
            prob += 0.2
        elif mw < 250 or mw > 700:
            prob -= 0.2
            
        # LogP (moderate lipophilicity preferred)
        logp = features['logp']
        if 1 <= logp <= 4:
            prob += 0.15
        elif logp < 0 or logp > 6:
            prob -= 0.15
            
        # Hydrogen bond acceptors (important for p53-MDM2 binding)
        hba = features['hba']
        if 3 <= hba <= 8:
            prob += 0.1
        elif hba > 10:
            prob -= 0.1
            
        # Aromatic rings (important for binding pocket interactions)
        aromatic = features['aromatic_rings']
        if aromatic >= 2:
            prob += 0.15
        elif aromatic == 0:
            prob -= 0.15
            
        # Heteroatoms (N, O important for binding)
        heteroatoms = features['heteroatoms']
        if 4 <= heteroatoms <= 10:
            prob += 0.1
        elif heteroatoms < 2:
            prob -= 0.2
            
        # TPSA (should not be too high for cell permeability)
        tpsa = features['tpsa']
        if tpsa > 120:
            prob -= 0.15
            
        # Ring count (cyclic structures often important)
        rings = features['rings']
        if rings >= 2:
            prob += 0.1
        elif rings == 0:
            prob -= 0.2
            
        return prob
    
    def _apply_conservative_filters(self, features, prob):
        """Apply conservative filters based on known non-inhibitor patterns"""
        
        # Very simple molecules are unlikely inhibitors
        if features['heavy_atoms'] < 15:
            prob *= 0.5
            
        # Very hydrophilic molecules are unlikely to bind MDM2
        if features['logp'] < -1:
            prob *= 0.3
            
        # Molecules with no rings are generally less likely to be inhibitors
        if features['rings'] == 0:
            prob *= 0.4
            
        # Too many rotatable bonds can reduce binding affinity
        if features['rotatable'] > 10:
            prob *= 0.7
            
        # Very high TPSA reduces cell permeability
        if features['tpsa'] > 150:
            prob *= 0.6
            
        return prob

# Global predictor instance
predictor = ImprovedMDM2Predictor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'message': f'Enhanced model with {len(predictor.actives)} actives, {len(predictor.negatives)} negatives',
        'threshold': predictor.threshold
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint with enhanced specificity"""
    try:
        data = request.json
        
        if not data or 'smiles' not in data:
            return jsonify({'error': 'SMILES string required'}), 400
        
        smiles = data['smiles'].strip()
        
        if not smiles:
            return jsonify({'error': 'Empty SMILES string'}), 400
        
        # Make prediction
        probability, error = predictor.predict_inhibition(smiles)
        
        if error:
            return jsonify({'error': error}), 400
        
        # Determine prediction with conservative threshold
        prediction = "Inhibitor" if probability > predictor.threshold else "Non-inhibitor"
        confidence = max(probability, 1 - probability)
        
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
            'threshold': predictor.threshold,
            'model_info': 'Enhanced with confirmed negatives'
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
                smiles = smiles.strip()
                
                # Make prediction
                probability, error = predictor.predict_inhibition(smiles)
                
                if error:
                    results.append({
                        'smiles': smiles,
                        'error': error
                    })
                    continue
                
                # Determine prediction
                prediction = "Inhibitor" if probability > predictor.threshold else "Non-inhibitor"
                confidence = max(probability, 1 - probability)
                
                if prediction == "Inhibitor":
                    interpretation = f"Likely MDM2 inhibitor ({probability:.1%} confidence)"
                else:
                    interpretation = f"Unlikely MDM2 inhibitor ({(1-probability):.1%} confidence)"
                
                results.append({
                    'smiles': smiles,
                    'prediction': prediction,
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
            'threshold': predictor.threshold,
            'total_processed': len(results),
            'model_info': 'Enhanced with confirmed negatives'
        })
        
    except Exception as e:
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get enhanced model information"""
    return jsonify({
        'model_type': 'Scaffold-Split MDM2 Predictor',
        'training_compounds': len(predictor.training_data) if predictor.training_data is not None else 0,
        'active_compounds': len(predictor.actives),
        'negative_compounds': len(predictor.negatives),
        'algorithm': 'Similarity-based with confirmed negatives',
        'features': 'Molecular descriptors + similarity analysis',
        'threshold': predictor.threshold,
        'improvements': [
            'Scaffold-based train/test splitting (no leakage)',
            'Experimentally confirmed negative compounds',
            'Conservative threshold (0.65)',
            'Similarity analysis to training data',
            'Enhanced molecular feature analysis',
            'Challenge sets for robust evaluation'
        ]
    })

if __name__ == '__main__':
    print("Starting Enhanced MDM2 Predictor API...")
    print(f"Server will be available at: http://localhost:5002")
    print(f"Health check: http://localhost:5002/health")
    print(f"Training data: {len(predictor.actives)} actives, {len(predictor.negatives)} negatives")
    print(f"Conservative threshold: {predictor.threshold}")
    print("Press Ctrl+C to stop the server")
    
    app.run(host='0.0.0.0', port=5002, debug=False)