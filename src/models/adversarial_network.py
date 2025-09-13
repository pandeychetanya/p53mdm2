"""
Adversarial Network for Enhanced MDM2 Inhibition Prediction
Implements Domain Adversarial Training for improved generalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GradientReverseFunction(Function):
    """
    Gradient Reverse Layer for adversarial training
    Reverses gradient during backpropagation
    """
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReverseLayer(nn.Module):
    """Gradient reversal layer wrapper"""
    
    def __init__(self, alpha=1.0):
        super(GradientReverseLayer, self).__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return GradientReverseFunction.apply(x, self.alpha)
    
    def set_alpha(self, alpha):
        self.alpha = alpha

class DomainDiscriminator(nn.Module):
    """
    Domain discriminator network
    Tries to distinguish between different molecular domains/datasets
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_domains: int = 2):
        super(DomainDiscriminator, self).__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_domains),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.discriminator(x)

class FeatureExtractor(nn.Module):
    """
    Feature extractor network
    Extracts domain-invariant features for classification
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super(FeatureExtractor, self).__init__()
        
        self.extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.extractor(x)

class TaskClassifier(nn.Module):
    """
    Task classifier network
    Performs MDM2 inhibition classification
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(TaskClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.classifier(x).squeeze()

class AdversarialMDM2Network(nn.Module):
    """
    Complete adversarial network for MDM2 inhibition prediction
    Combines feature extraction, task classification, and domain discrimination
    """
    
    def __init__(self, 
                 input_dim: int,
                 feature_dim: int = 128,
                 hidden_dim: int = 256,
                 num_domains: int = 2,
                 alpha: float = 1.0):
        super(AdversarialMDM2Network, self).__init__()
        
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dim, feature_dim)
        self.task_classifier = TaskClassifier(feature_dim)
        self.gradient_reverse = GradientReverseLayer(alpha)
        self.domain_discriminator = DomainDiscriminator(feature_dim, hidden_dim // 2, num_domains)
        
        self.alpha = alpha
    
    def forward(self, x, return_features: bool = False):
        # Extract features
        features = self.feature_extractor(x)
        
        # Task prediction (MDM2 inhibition)
        task_pred = self.task_classifier(features)
        
        # Domain prediction (with gradient reversal)
        reversed_features = self.gradient_reverse(features)
        domain_pred = self.domain_discriminator(reversed_features)
        
        if return_features:
            return task_pred, domain_pred, features
        else:
            return task_pred, domain_pred
    
    def predict(self, x):
        """Make predictions without domain discrimination"""
        with torch.no_grad():
            features = self.feature_extractor(x)
            task_pred = self.task_classifier(features)
            return task_pred
    
    def set_alpha(self, alpha: float):
        """Update gradient reversal strength"""
        self.alpha = alpha
        self.gradient_reverse.set_alpha(alpha)

class AdversarialTrainer:
    """
    Trainer for adversarial MDM2 network
    """
    
    def __init__(self, 
                 model: AdversarialMDM2Network,
                 device: str = 'cpu',
                 lr: float = 0.001,
                 lambda_domain: float = 0.1):
        
        self.model = model.to(device)
        self.device = device
        self.lambda_domain = lambda_domain
        
        # Loss functions
        self.task_criterion = nn.BCELoss()
        self.domain_criterion = nn.CrossEntropyLoss()
        
        # Optimizers
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10, factor=0.5
        )
    
    def train_step(self, 
                   x: torch.Tensor, 
                   y_task: torch.Tensor,
                   y_domain: torch.Tensor,
                   alpha: float = 1.0) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            x: Input features
            y_task: Task labels (MDM2 inhibition)
            y_domain: Domain labels
            alpha: Gradient reversal strength
            
        Returns:
            Dictionary of losses
        """
        self.model.train()
        self.model.set_alpha(alpha)
        
        x = x.to(self.device)
        y_task = y_task.to(self.device)
        y_domain = y_domain.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass
        task_pred, domain_pred = self.model(x)
        
        # Calculate losses
        task_loss = self.task_criterion(task_pred, y_task)
        domain_loss = self.domain_criterion(domain_pred, y_domain)
        
        # Combined loss
        total_loss = task_loss + self.lambda_domain * domain_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'task_loss': task_loss.item(),
            'domain_loss': domain_loss.item(),
            'alpha': alpha
        }
    
    def evaluate(self, 
                 x: torch.Tensor,
                 y_task: torch.Tensor,
                 y_domain: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            x: Input features
            y_task: Task labels
            y_domain: Domain labels (optional)
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            x = x.to(self.device)
            y_task = y_task.to(self.device)
            
            if y_domain is not None:
                y_domain = y_domain.to(self.device)
                task_pred, domain_pred = self.model(x)
                domain_loss = self.domain_criterion(domain_pred, y_domain).item()
                domain_acc = (domain_pred.argmax(dim=1) == y_domain).float().mean().item()
            else:
                task_pred = self.model.predict(x)
                domain_loss = 0.0
                domain_acc = 0.0
            
            # Task metrics
            task_loss = self.task_criterion(task_pred, y_task).item()
            binary_pred = (task_pred > 0.5).float()
            task_acc = (binary_pred == y_task).float().mean().item()
            
            # Confidence metrics
            confidence_scores = torch.abs(task_pred - 0.5) * 2  # Convert to 0-1 range
            avg_confidence = confidence_scores.mean().item()
            
        return {
            'task_loss': task_loss,
            'task_accuracy': task_acc,
            'domain_loss': domain_loss,
            'domain_accuracy': domain_acc,
            'avg_confidence': avg_confidence,
            'predictions': task_pred.cpu().numpy(),
            'confidence_scores': confidence_scores.cpu().numpy()
        }
    
    def get_alpha_schedule(self, epoch: int, max_epochs: int) -> float:
        """
        Dynamic alpha scheduling for gradient reversal
        Gradually increases adversarial strength during training
        """
        p = float(epoch) / max_epochs
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        return alpha

def create_synthetic_domain_data(features: np.ndarray, 
                                labels: np.ndarray,
                                domain_split_ratio: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic domain labels for adversarial training
    
    Args:
        features: Feature matrix
        labels: Task labels
        domain_split_ratio: Ratio for domain split
        
    Returns:
        Tuple of (domain_labels, domain_weights)
    """
    n_samples = len(features)
    n_domain1 = int(n_samples * domain_split_ratio)
    
    # Create domain labels (0 for domain 1, 1 for domain 2)
    domain_labels = np.zeros(n_samples, dtype=int)
    domain_labels[n_domain1:] = 1
    
    # Add some class imbalance to make it more realistic
    np.random.shuffle(domain_labels)
    
    return domain_labels

def main():
    """Test adversarial network implementation"""
    # Create synthetic data
    n_samples = 100
    n_features = 7  # From evolutionary feature selection
    
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y_task = np.random.randint(0, 2, n_samples)
    y_domain = create_synthetic_domain_data(X, y_task)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_task_tensor = torch.FloatTensor(y_task)
    y_domain_tensor = torch.LongTensor(y_domain)
    
    # Create model
    model = AdversarialMDM2Network(
        input_dim=n_features,
        feature_dim=64,
        hidden_dim=128,
        num_domains=2,
        alpha=0.1
    )
    
    # Create trainer
    trainer = AdversarialTrainer(model, lambda_domain=0.1)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test training step
    losses = trainer.train_step(X_tensor, y_task_tensor, y_domain_tensor, alpha=0.5)
    print(f"Training losses: {losses}")
    
    # Test evaluation
    metrics = trainer.evaluate(X_tensor, y_task_tensor, y_domain_tensor)
    print(f"Evaluation metrics: {metrics}")
    
    # Test prediction
    with torch.no_grad():
        predictions = model.predict(X_tensor[:10])
        print(f"Sample predictions: {predictions[:5].tolist()}")
    
    print("Adversarial network test completed successfully!")

if __name__ == "__main__":
    main()