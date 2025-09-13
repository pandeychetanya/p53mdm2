"""
Hybrid MDM2 Prediction Model
Combines Graph Neural Networks with Adversarial Networks and molecular descriptors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader as GeometricDataLoader
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import logging

# Import custom modules
from molecular_gnn import MolecularGNN, MolecularGraphBuilder
from adversarial_network import AdversarialMDM2Network

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridMDM2Model(nn.Module):
    """
    Hybrid model combining GNN and adversarial networks for MDM2 inhibition prediction
    """
    
    def __init__(self,
                 gnn_config: Dict,
                 adversarial_config: Dict,
                 descriptor_dim: int,
                 fusion_dim: int = 128,
                 dropout: float = 0.3):
        super(HybridMDM2Model, self).__init__()
        
        self.descriptor_dim = descriptor_dim
        self.fusion_dim = fusion_dim
        
        # GNN component for molecular graph analysis
        self.gnn = MolecularGNN(**gnn_config)
        
        # Adversarial network for descriptor-based prediction
        self.adversarial_net = AdversarialMDM2Network(**adversarial_config)
        
        # Feature fusion layers
        self.gnn_projection = nn.Linear(gnn_config.get('hidden_dim', 128) * 8, fusion_dim)  # *8 for GAT with 4 heads + pooling
        self.descriptor_projection = nn.Linear(adversarial_config['feature_dim'], fusion_dim)
        
        # Fusion mechanism
        self.attention = nn.MultiheadAttention(fusion_dim, num_heads=4, dropout=dropout)
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        
        # Final classifier
        self.final_classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),  # *2 for concatenated features
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, graph_data, descriptor_data, return_confidence=False):
        """
        Forward pass through hybrid model
        
        Args:
            graph_data: PyTorch Geometric batch for GNN
            descriptor_data: Tensor of molecular descriptors
            return_confidence: Whether to return confidence scores
        """
        batch_size = descriptor_data.shape[0]
        
        # GNN pathway
        # Extract features before final classification
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        
        # Run through GNN layers
        gnn_features = x
        for i, (conv, bn) in enumerate(zip(self.gnn.convs, self.gnn.batch_norms)):
            gnn_features = conv(gnn_features, edge_index)
            gnn_features = bn(gnn_features)
            gnn_features = F.relu(gnn_features)
            if i < self.gnn.num_layers - 1:
                gnn_features = self.gnn.dropout(gnn_features)
        
        # Global pooling
        from torch_geometric.nn import global_mean_pool, global_max_pool
        gnn_mean = global_mean_pool(gnn_features, batch)
        gnn_max = global_max_pool(gnn_features, batch)
        gnn_pooled = torch.cat([gnn_mean, gnn_max], dim=1)
        
        # Project GNN features
        gnn_projected = self.gnn_projection(gnn_pooled)
        
        # Adversarial network pathway (without domain discrimination for fusion)
        desc_features = self.adversarial_net.feature_extractor(descriptor_data)
        desc_projected = self.descriptor_projection(desc_features)
        
        # Feature fusion with attention
        # Reshape for attention: (seq_len, batch_size, feature_dim)
        gnn_seq = gnn_projected.unsqueeze(0)  # (1, batch_size, fusion_dim)
        desc_seq = desc_projected.unsqueeze(0)  # (1, batch_size, fusion_dim)
        
        # Self-attention over both feature types
        combined_seq = torch.cat([gnn_seq, desc_seq], dim=0)  # (2, batch_size, fusion_dim)
        attended_features, attention_weights = self.attention(combined_seq, combined_seq, combined_seq)
        
        # Normalize and aggregate
        attended_features = self.fusion_norm(attended_features)
        fused_features = attended_features.mean(dim=0)  # Average over sequence dimension
        
        # Concatenate original projections with fused features
        final_features = torch.cat([gnn_projected, desc_projected], dim=1)
        
        # Final prediction
        prediction = self.final_classifier(final_features)
        
        if return_confidence:
            confidence = self.confidence_estimator(final_features)
            return prediction.squeeze(), confidence.squeeze()
        else:
            return prediction.squeeze()
    
    def predict_with_confidence(self, graph_data, descriptor_data):
        """Make prediction with confidence score"""
        self.eval()
        with torch.no_grad():
            prediction, confidence = self.forward(graph_data, descriptor_data, return_confidence=True)
            return prediction, confidence

class HybridTrainer:
    """Trainer for hybrid MDM2 model"""
    
    def __init__(self, model: HybridMDM2Model, device: str = 'cpu', lr: float = 0.001):
        self.model = model.to(device)
        self.device = device
        
        # Loss functions
        self.prediction_criterion = nn.BCELoss()
        self.confidence_criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=15, factor=0.5
        )
        
    def train_step(self, graph_batch, descriptor_batch, labels) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        graph_batch = graph_batch.to(self.device)
        descriptor_batch = descriptor_batch.to(self.device)
        labels = labels.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions, confidences = self.model.forward(
            graph_batch, descriptor_batch, return_confidence=True
        )
        
        # Calculate losses
        pred_loss = self.prediction_criterion(predictions, labels)
        
        # Confidence target: higher confidence for correct predictions
        correct_predictions = (predictions.round() == labels).float()
        conf_loss = self.confidence_criterion(confidences, correct_predictions)
        
        # Combined loss
        total_loss = pred_loss + 0.1 * conf_loss
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'prediction_loss': pred_loss.item(),
            'confidence_loss': conf_loss.item()
        }
    
    def evaluate(self, graph_batch, descriptor_batch, labels) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        
        with torch.no_grad():
            graph_batch = graph_batch.to(self.device)
            descriptor_batch = descriptor_batch.to(self.device)
            labels = labels.to(self.device)
            
            predictions, confidences = self.model.predict_with_confidence(
                graph_batch, descriptor_batch
            )
            
            # Metrics
            pred_loss = self.prediction_criterion(predictions, labels).item()
            binary_preds = (predictions > 0.5).float()
            accuracy = (binary_preds == labels).float().mean().item()
            
            # F1 Score
            tp = ((binary_preds == 1) & (labels == 1)).float().sum()
            fp = ((binary_preds == 1) & (labels == 0)).float().sum()
            fn = ((binary_preds == 0) & (labels == 1)).float().sum()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
            
        return {
            'loss': pred_loss,
            'accuracy': accuracy,
            'f1_score': f1_score.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'avg_confidence': confidences.mean().item(),
            'predictions': predictions.cpu().numpy(),
            'confidences': confidences.cpu().numpy()
        }

def create_hybrid_model(descriptor_dim: int = 7) -> HybridMDM2Model:
    """Create hybrid model with default configurations"""
    
    # GNN configuration
    gnn_config = {
        'num_atom_features': 135,  # From molecular graph builder
        'hidden_dim': 64,
        'num_layers': 3,
        'dropout': 0.2,
        'use_attention': True
    }
    
    # Adversarial network configuration
    adversarial_config = {
        'input_dim': descriptor_dim,
        'feature_dim': 64,
        'hidden_dim': 128,
        'num_domains': 2,
        'alpha': 0.1
    }
    
    model = HybridMDM2Model(
        gnn_config=gnn_config,
        adversarial_config=adversarial_config,
        descriptor_dim=descriptor_dim,
        fusion_dim=128,
        dropout=0.3
    )
    
    return model

def main():
    """Test hybrid model implementation"""
    # Create synthetic data
    from torch_geometric.data import Data, Batch
    
    # Mock graph data (representing molecular graphs)
    num_graphs = 8
    graphs = []
    
    for i in range(num_graphs):
        num_nodes = np.random.randint(10, 20)
        num_edges = np.random.randint(15, 30)
        
        x = torch.randn(num_nodes, 135)  # Node features
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        y = torch.tensor([np.random.randint(0, 2)], dtype=torch.float)
        
        graphs.append(Data(x=x, edge_index=edge_index, y=y))
    
    graph_batch = Batch.from_data_list(graphs)
    
    # Mock descriptor data
    descriptor_data = torch.randn(num_graphs, 7)
    labels = torch.tensor([graph.y.item() for graph in graphs], dtype=torch.float)
    
    # Create model
    model = create_hybrid_model(descriptor_dim=7)
    
    print(f"Hybrid model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    with torch.no_grad():
        predictions = model.forward(graph_batch, descriptor_data)
        print(f"Predictions shape: {predictions.shape}")
        print(f"Sample predictions: {predictions[:5].tolist()}")
        
        # Test with confidence
        pred_with_conf, confidences = model.predict_with_confidence(graph_batch, descriptor_data)
        print(f"Predictions with confidence: {list(zip(pred_with_conf[:3].tolist(), confidences[:3].tolist()))}")
    
    # Test training step
    trainer = HybridTrainer(model)
    losses = trainer.train_step(graph_batch, descriptor_data, labels)
    print(f"Training losses: {losses}")
    
    # Test evaluation
    metrics = trainer.evaluate(graph_batch, descriptor_data, labels)
    print(f"Evaluation metrics: {metrics}")
    
    print("Hybrid model test completed successfully!")

if __name__ == "__main__":
    main()