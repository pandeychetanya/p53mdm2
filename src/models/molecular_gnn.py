"""
Graph Neural Network for Molecular Structure Analysis
Processes SMILES strings as molecular graphs for MDM2 inhibition prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import rdchem
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MolecularGraphBuilder:
    """Convert SMILES to molecular graphs for GNN processing"""
    
    def __init__(self):
        # Atom feature dimensions
        self.atom_features = {
            'atomic_num': 100,  # atomic number
            'degree': 10,       # node degree
            'formal_charge': 11,  # formal charge (-5 to +5)
            'hybridization': 8,   # hybridization type
            'aromatic': 2,        # aromatic (binary)
            'chirality': 4        # chirality
        }
        self.num_atom_features = sum(self.atom_features.values())
        
        # Bond feature dimensions
        self.bond_features = {
            'bond_type': 4,     # single, double, triple, aromatic
            'conjugated': 2,    # conjugated (binary)
            'in_ring': 2        # in ring (binary)
        }
        self.num_bond_features = sum(self.bond_features.values())
    
    def get_atom_features(self, atom) -> List[int]:
        """Extract atom features"""
        features = []
        
        # Atomic number (one-hot up to 100)
        atomic_num = atom.GetAtomicNum()
        atomic_features = [0] * self.atom_features['atomic_num']
        if atomic_num < self.atom_features['atomic_num']:
            atomic_features[atomic_num] = 1
        features.extend(atomic_features)
        
        # Degree (one-hot)
        degree = atom.GetDegree()
        degree_features = [0] * self.atom_features['degree']
        if degree < self.atom_features['degree']:
            degree_features[degree] = 1
        features.extend(degree_features)
        
        # Formal charge (offset by 5)
        formal_charge = atom.GetFormalCharge()
        charge_features = [0] * self.atom_features['formal_charge']
        if -5 <= formal_charge <= 5:
            charge_features[formal_charge + 5] = 1
        features.extend(charge_features)
        
        # Hybridization
        hybridization = atom.GetHybridization()
        hyb_features = [0] * self.atom_features['hybridization']
        hyb_map = {
            rdchem.HybridizationType.SP: 0,
            rdchem.HybridizationType.SP2: 1,
            rdchem.HybridizationType.SP3: 2,
            rdchem.HybridizationType.SP3D: 3,
            rdchem.HybridizationType.SP3D2: 4
        }
        if hybridization in hyb_map:
            hyb_features[hyb_map[hybridization]] = 1
        features.extend(hyb_features)
        
        # Aromatic
        aromatic_features = [1, 0] if atom.GetIsAromatic() else [0, 1]
        features.extend(aromatic_features)
        
        # Chirality
        chirality = atom.GetChiralTag()
        chiral_features = [0] * self.atom_features['chirality']
        chiral_map = {
            rdchem.ChiralType.CHI_UNSPECIFIED: 0,
            rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
            rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
            rdchem.ChiralType.CHI_OTHER: 3
        }
        if chirality in chiral_map:
            chiral_features[chiral_map[chirality]] = 1
        features.extend(chiral_features)
        
        return features
    
    def get_bond_features(self, bond) -> List[int]:
        """Extract bond features"""
        features = []
        
        # Bond type
        bond_type = bond.GetBondType()
        type_features = [0] * self.bond_features['bond_type']
        type_map = {
            rdchem.BondType.SINGLE: 0,
            rdchem.BondType.DOUBLE: 1,
            rdchem.BondType.TRIPLE: 2,
            rdchem.BondType.AROMATIC: 3
        }
        if bond_type in type_map:
            type_features[type_map[bond_type]] = 1
        features.extend(type_features)
        
        # Conjugated
        conjugated_features = [1, 0] if bond.GetIsConjugated() else [0, 1]
        features.extend(conjugated_features)
        
        # In ring
        in_ring_features = [1, 0] if bond.IsInRing() else [0, 1]
        features.extend(in_ring_features)
        
        return features
    
    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """Convert SMILES to PyTorch Geometric Data object"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return None
            
            # Add hydrogens for complete graph
            mol = Chem.AddHs(mol)
            
            # Atom features
            atom_features = []
            for atom in mol.GetAtoms():
                features = self.get_atom_features(atom)
                atom_features.append(features)
            
            # Edge indices and features
            edge_indices = []
            edge_features = []
            
            for bond in mol.GetBonds():
                start_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                
                # Add both directions for undirected graph
                edge_indices.extend([[start_idx, end_idx], [end_idx, start_idx]])
                
                # Bond features
                bond_feat = self.get_bond_features(bond)
                edge_features.extend([bond_feat, bond_feat])
            
            # Convert to tensors
            x = torch.tensor(atom_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float) if edge_features else None
            
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            
        except Exception as e:
            logger.error(f"Error converting SMILES {smiles}: {e}")
            return None
    
    def create_dataset(self, smiles_list: List[str], 
                      labels: Optional[List[int]] = None) -> List[Data]:
        """Create dataset of molecular graphs"""
        dataset = []
        
        for i, smiles in enumerate(smiles_list):
            if i % 100 == 0:
                logger.info(f"Processing molecule {i}/{len(smiles_list)}")
            
            graph = self.smiles_to_graph(smiles)
            if graph is not None:
                if labels is not None:
                    graph.y = torch.tensor([labels[i]], dtype=torch.float)
                dataset.append(graph)
        
        logger.info(f"Created dataset with {len(dataset)} valid graphs")
        return dataset

class MolecularGNN(nn.Module):
    """Graph Neural Network for molecular property prediction"""
    
    def __init__(self, 
                 num_atom_features: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 use_attention: bool = True):
        super(MolecularGNN, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        if use_attention:
            self.convs.append(GATConv(num_atom_features, hidden_dim, heads=4, dropout=dropout))
        else:
            self.convs.append(GCNConv(num_atom_features, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim * (4 if use_attention else 1)))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            if use_attention:
                self.convs.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=dropout))
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * (4 if use_attention else 1)))
        
        # Global pooling and classification
        pooled_dim = hidden_dim * (4 if use_attention else 1)
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim * 2, hidden_dim),  # *2 for mean+max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph convolutions
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i < self.num_layers - 1:  # No dropout on last layer
                x = self.dropout(x)
        
        # Global pooling (combine mean and max pooling)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # Classification
        out = self.classifier(x)
        return out.squeeze()

class GNNTrainer:
    """Trainer for molecular GNN"""
    
    def __init__(self, model: MolecularGNN, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10, factor=0.5, verbose=True
        )
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            out = self.model(batch)
            loss = self.criterion(out, batch.y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                out = self.model(batch)
                loss = self.criterion(out, batch.y)
                
                total_loss += loss.item()
                predictions.extend(out.cpu().numpy())
                targets.extend(batch.y.cpu().numpy())
        
        # Convert to binary predictions
        binary_preds = (np.array(predictions) > 0.5).astype(int)
        targets = np.array(targets)
        
        accuracy = (binary_preds == targets).mean()
        
        return {
            'loss': total_loss / len(test_loader),
            'accuracy': accuracy,
            'predictions': predictions,
            'targets': targets
        }

def main():
    """Test GNN implementation"""
    # Load test data
    data = pd.read_csv("../../data/raw/mdm2_test_data.csv")
    
    # Create molecular graphs
    graph_builder = MolecularGraphBuilder()
    
    # Use subset for testing
    smiles_list = data['canonical_smiles'].dropna().tolist()[:10]
    labels = data['is_inhibitor'].iloc[:10].tolist()
    
    dataset = graph_builder.create_dataset(smiles_list, labels)
    
    if len(dataset) > 0:
        # Create model
        model = MolecularGNN(
            num_atom_features=graph_builder.num_atom_features,
            hidden_dim=64,
            num_layers=2,
            use_attention=True
        )
        
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        print(f"Dataset size: {len(dataset)} graphs")
        print(f"Feature dimensions: {graph_builder.num_atom_features} atom features")
        
        # Test forward pass
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        
        with torch.no_grad():
            output = model(batch)
            print(f"Output shape: {output.shape}")
            print(f"Sample predictions: {output[:5].tolist()}")
    else:
        print("No valid graphs created from SMILES")

if __name__ == "__main__":
    main()