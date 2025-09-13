#!/usr/bin/env python3
"""
Start the MDM2 Predictor API server
"""

import sys
import os
import subprocess
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = ['flask', 'flask_cors', 'rdkit', 'scikit-learn', 'pandas', 'numpy', 'imbalanced-learn']
    missing = []
    
    for package in required_packages:
        try:
            if package == 'flask_cors':
                import flask_cors
            elif package == 'rdkit':
                from rdkit import Chem
            elif package == 'scikit-learn':
                import sklearn
            elif package == 'imbalanced-learn':
                import imblearn
            else:
                __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True

def check_data():
    """Check if training data exists"""
    data_path = Path(__file__).parent / 'data' / 'processed' / 'balanced_mdm2_data.csv'
    
    if not data_path.exists():
        print(f"Training data not found at: {data_path}")
        print("Please run data collection first")
        return False
    
    # Check data size
    try:
        data = pd.read_csv(data_path)
        print(f"Found training data: {len(data)} compounds")
        return True
    except Exception as e:
        print(f"Error reading training data: {e}")
        return False

def start_server():
    """Start the Flask API server"""
    if not check_dependencies():
        return False
    
    if not check_data():
        return False
    
    print("Starting MDM2 Predictor API server...")
    print("Server will be available at: http://localhost:5000")
    print("Health check: http://localhost:5000/health")
    print("Press Ctrl+C to stop the server")
    
    # Start Flask app
    api_path = Path(__file__).parent / 'api' / 'app.py'
    subprocess.run([sys.executable, str(api_path)])

if __name__ == '__main__':
    start_server()