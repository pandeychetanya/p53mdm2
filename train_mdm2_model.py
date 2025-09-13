"""
Complete MDM2 Inhibition Prediction Model Training Script
Trains the hybrid GNN-Adversarial model on ChEMBL data with evolutionary feature selection
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import logging
import argparse
from pathlib import Path

# Import custom modules
import sys
sys.path.append('src/models')
sys.path.append('src/data')
sys.path.append('src/features')

from prediction_pipeline import MDM2InhibitionPredictor, format_prediction_output
from hybrid_model import HybridTrainer, create_hybrid_model
from molecular_gnn import MolecularGraphBuilder
from torch_geometric.data import DataLoader as GeometricDataLoader, Batch

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model_performance(y_true, y_pred, y_prob):
    """Calculate comprehensive evaluation metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0
    }
    return metrics

def train_mdm2_model(data_path: str, 
                     epochs: int = 50, 
                     cross_validation: bool = True,
                     cv_folds: int = 5,
                     save_model: bool = True):
    """
    Train MDM2 inhibition prediction model
    
    Args:
        data_path: Path to MDM2 dataset CSV
        epochs: Number of training epochs
        cross_validation: Whether to use cross-validation
        cv_folds: Number of CV folds
        save_model: Whether to save the trained model
    """
    logger.info("Starting MDM2 model training...")
    
    # Initialize predictor
    predictor = MDM2InhibitionPredictor()
    
    # Setup pipeline with feature selection
    X_train, X_test, y_train, y_test = predictor.setup_pipeline(
        data_path, 
        feature_selection=True, 
        save_components=True
    )
    
    logger.info(f"Dataset: Train {len(X_train)}, Test {len(X_test)}")
    logger.info(f"Selected features ({len(predictor.selected_features)}): {predictor.selected_features}")
    
    if cross_validation:
        # Cross-validation training
        logger.info(f"Starting {cv_folds}-fold cross-validation...")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_results = []
        
        # Load data for CV
        df = pd.read_csv(data_path)
        valid_indices = df['canonical_smiles'].notna()
        df_valid = df[valid_indices].reset_index(drop=True)
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(df_valid, df_valid['is_inhibitor'])):
            logger.info(f"Training fold {fold + 1}/{cv_folds}")
            
            # Prepare fold data
            train_smiles = df_valid.iloc[train_idx]['canonical_smiles'].tolist()
            val_smiles = df_valid.iloc[val_idx]['canonical_smiles'].tolist()
            train_labels = df_valid.iloc[train_idx]['is_inhibitor'].values
            val_labels = df_valid.iloc[val_idx]['is_inhibitor'].values
            
            # Create graphs for this fold
            graph_builder = MolecularGraphBuilder()
            train_graphs = graph_builder.create_dataset(train_smiles, train_labels.tolist())
            val_graphs = graph_builder.create_dataset(val_smiles, val_labels.tolist())
            
            if len(train_graphs) == 0 or len(val_graphs) == 0:
                logger.warning(f"Skipping fold {fold + 1} - insufficient valid graphs")
                continue
            
            # Prepare descriptors
            train_descriptors = []
            val_descriptors = []
            
            for idx in train_idx:
                try:
                    features = predictor.data_processor.process_new_smiles(
                        df_valid.iloc[idx]['canonical_smiles'], use_rdkit=False
                    )
                    train_descriptors.append(features)
                except:
                    train_descriptors.append(np.zeros(len(predictor.selected_features)))
            
            for idx in val_idx:
                try:
                    features = predictor.data_processor.process_new_smiles(
                        df_valid.iloc[idx]['canonical_smiles'], use_rdkit=False
                    )
                    val_descriptors.append(features)
                except:
                    val_descriptors.append(np.zeros(len(predictor.selected_features)))
            
            # Train fold model
            fold_model = predictor.model
            trainer = HybridTrainer(fold_model, device=predictor.device)
            
            # Training loop for this fold
            best_val_f1 = 0.0
            patience_counter = 0
            patience = 15
            
            for epoch in range(min(epochs, 30)):  # Limit epochs for CV
                # Create batches
                train_batch = Batch.from_data_list(train_graphs[:len(train_descriptors)])
                train_desc_tensor = torch.FloatTensor(train_descriptors[:len(train_graphs)])
                train_labels_tensor = torch.FloatTensor([g.y.item() for g in train_graphs[:len(train_descriptors)]])
                
                # Training step
                train_losses = trainer.train_step(train_batch, train_desc_tensor, train_labels_tensor)
                
                # Validation every 5 epochs
                if epoch % 5 == 0:
                    val_batch = Batch.from_data_list(val_graphs[:len(val_descriptors)])
                    val_desc_tensor = torch.FloatTensor(val_descriptors[:len(val_graphs)])
                    val_labels_tensor = torch.FloatTensor([g.y.item() for g in val_graphs[:len(val_descriptors)]])
                    
                    val_metrics = trainer.evaluate(val_batch, val_desc_tensor, val_labels_tensor)
                    
                    if val_metrics['f1_score'] > best_val_f1:
                        best_val_f1 = val_metrics['f1_score']
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
            
            # Final validation metrics
            val_batch = Batch.from_data_list(val_graphs[:len(val_descriptors)])
            val_desc_tensor = torch.FloatTensor(val_descriptors[:len(val_graphs)])
            val_labels_tensor = torch.FloatTensor([g.y.item() for g in val_graphs[:len(val_descriptors)]])
            
            val_metrics = trainer.evaluate(val_batch, val_desc_tensor, val_labels_tensor)
            cv_results.append(val_metrics)
            
            logger.info(f"Fold {fold + 1} - Val F1: {val_metrics['f1_score']:.3f}, "
                       f"Accuracy: {val_metrics['accuracy']:.3f}")
        
        # Calculate CV statistics
        if cv_results:
            cv_stats = {}
            for metric in ['accuracy', 'f1_score', 'precision', 'recall']:
                values = [r[metric] for r in cv_results]
                cv_stats[f'{metric}_mean'] = np.mean(values)
                cv_stats[f'{metric}_std'] = np.std(values)
            
            logger.info("Cross-Validation Results:")
            for metric, value in cv_stats.items():
                logger.info(f"{metric}: {value:.4f}")
        
    # Final training on full dataset
    logger.info("Training final model on full dataset...")
    
    # Create full training dataset
    df = pd.read_csv(data_path)
    valid_indices = df['canonical_smiles'].notna()
    df_valid = df[valid_indices].reset_index(drop=True)
    
    # Create graphs
    graph_builder = MolecularGraphBuilder()
    all_graphs = graph_builder.create_dataset(
        df_valid['canonical_smiles'].tolist(),
        df_valid['is_inhibitor'].tolist()
    )
    
    # Prepare descriptors
    all_descriptors = []
    for _, row in df_valid.iterrows():
        try:
            features = predictor.data_processor.process_new_smiles(
                row['canonical_smiles'], use_rdkit=False
            )
            all_descriptors.append(features)
        except:
            all_descriptors.append(np.zeros(len(predictor.selected_features)))
    
    # Split for final training
    split_idx = int(0.8 * len(all_graphs))
    train_graphs = all_graphs[:split_idx]
    test_graphs = all_graphs[split_idx:]
    train_descriptors = all_descriptors[:split_idx]
    test_descriptors = all_descriptors[split_idx:]
    
    # Final training
    trainer = HybridTrainer(predictor.model, device=predictor.device)
    
    best_test_f1 = 0.0
    for epoch in range(epochs):
        # Training
        if len(train_graphs) > 0 and len(train_descriptors) > 0:
            train_batch = Batch.from_data_list(train_graphs[:len(train_descriptors)])
            train_desc_tensor = torch.FloatTensor(train_descriptors[:len(train_graphs)])
            train_labels_tensor = torch.FloatTensor([g.y.item() for g in train_graphs[:len(train_descriptors)]])
            
            train_losses = trainer.train_step(train_batch, train_desc_tensor, train_labels_tensor)
        
        # Evaluation every 10 epochs
        if epoch % 10 == 0 and len(test_graphs) > 0:
            test_batch = Batch.from_data_list(test_graphs[:len(test_descriptors)])
            test_desc_tensor = torch.FloatTensor(test_descriptors[:len(test_graphs)])
            test_labels_tensor = torch.FloatTensor([g.y.item() for g in test_graphs[:len(test_descriptors)]])
            
            test_metrics = trainer.evaluate(test_batch, test_desc_tensor, test_labels_tensor)
            
            logger.info(f"Epoch {epoch:3d} - Loss: {train_losses['total_loss']:.4f}, "
                       f"Test F1: {test_metrics['f1_score']:.3f}, "
                       f"Accuracy: {test_metrics['accuracy']:.3f}")
            
            if test_metrics['f1_score'] > best_test_f1:
                best_test_f1 = test_metrics['f1_score']
                
                if save_model:
                    predictor.save_model("models/best_mdm2_model.pth")
    
    # Final evaluation
    if len(test_graphs) > 0:
        test_batch = Batch.from_data_list(test_graphs[:len(test_descriptors)])
        test_desc_tensor = torch.FloatTensor(test_descriptors[:len(test_graphs)])
        test_labels_tensor = torch.FloatTensor([g.y.item() for g in test_graphs[:len(test_descriptors)]])
        
        final_metrics = trainer.evaluate(test_batch, test_desc_tensor, test_labels_tensor)
        
        logger.info("Final Test Results:")
        for metric, value in final_metrics.items():
            if isinstance(value, float):
                logger.info(f"{metric}: {value:.4f}")
    
    logger.info("Training completed successfully!")
    return predictor

def demo_predictions():
    """Demonstrate predictions on example SMILES"""
    
    # Example drug-like molecules
    demo_smiles = [
        "CCO",  # Ethanol (simple)
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "C1=CC=C(C=C1)C(=O)O",  # Benzoic acid
        "CN1CCC[C@H]1C2=CN=CC=C2",  # Nicotine
        "CC1=CC=CC=C1C"  # Toluene
    ]
    
    logger.info("Running demonstration predictions...")
    
    try:
        # Create predictor with mock setup
        predictor = MDM2InhibitionPredictor()
        
        # Mock setup for demo
        from sklearn.preprocessing import StandardScaler
        
        predictor.selected_features = ['alogp', 'hba', 'hbd', 'rtb', 'num_ro5_violations', 'cx_logp', 'aromatic_rings']
        predictor.model = create_hybrid_model(descriptor_dim=7)
        predictor.model.eval()
        
        # Mock scaler
        predictor.scaler = StandardScaler()
        predictor.scaler.mean_ = np.zeros(7)
        predictor.scaler.scale_ = np.ones(7)
        predictor.data_processor.scaler = predictor.scaler
        predictor.data_processor.selected_features = predictor.selected_features
        
        # Make predictions
        results = predictor.predict_smiles(demo_smiles, return_confidence=True)
        
        # Format output
        formatted_output = format_prediction_output(results, detailed=True)
        print(formatted_output)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")

def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description="Train MDM2 Inhibition Prediction Model")
    parser.add_argument("--data", type=str, default="data/raw/mdm2_test_data.csv",
                       help="Path to MDM2 dataset CSV file")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--cv", action="store_true",
                       help="Use cross-validation")
    parser.add_argument("--cv_folds", type=int, default=5,
                       help="Number of CV folds")
    parser.add_argument("--save_model", action="store_true", default=True,
                       help="Save trained model")
    parser.add_argument("--demo", action="store_true",
                       help="Run demonstration predictions only")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_predictions()
        return
    
    # Check if data file exists
    if not Path(args.data).exists():
        logger.error(f"Data file not found: {args.data}")
        logger.info("Running demo instead...")
        demo_predictions()
        return
    
    try:
        # Train model
        predictor = train_mdm2_model(
            data_path=args.data,
            epochs=args.epochs,
            cross_validation=args.cv,
            cv_folds=args.cv_folds,
            save_model=args.save_model
        )
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.info("Running demo instead...")
        demo_predictions()

if __name__ == "__main__":
    main()