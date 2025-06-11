"""
Training Script for Face Track Linking Model

This script trains the face track linking model based on research insights
from our curated dataset of face tracking papers.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from face_track_linking_model import (
    FaceTrackLinkingModel,
    TripletLoss,
    ContrastiveLoss,
    TrackLinkingMetrics
)
# Import face recognition system (optional - will use synthetic data if not available)
try:
    from face_recognition_module import FaceRecognitionSystem
    FACE_RECOGNITION_AVAILABLE = True
    print("✅ Face recognition module loaded successfully")
except ImportError as e:
    print(f"⚠️ Face recognition module not available: {e}")
    print("📝 Will use synthetic data generation instead")
    FACE_RECOGNITION_AVAILABLE = False


class FaceTrackLinkingTrainer:
    """
    Trainer for Face Track Linking Model
    Implements training strategies based on research insights
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
          # Create model
        self.model = FaceTrackLinkingModel(
            backbone=config['model'].get('backbone', 'resnet50'),
            embedding_dim=config['model'].get('embedding_dim', 512),
            max_distance=config['model'].get('max_distance', 0.7)
        )
        self.model.to(self.device)
        
        # Create optimizers
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Create loss functions
        self.triplet_loss = TripletLoss(margin=config['training']['triplet_margin'])
        self.contrastive_loss = ContrastiveLoss(margin=config['training']['contrastive_margin'])
        self.mse_loss = nn.MSELoss()
        
        # Training state
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.train_losses = []
        self.val_accuracies = []
        
        # Logging
        self.writer = SummaryWriter(config['logging']['tensorboard_dir'])
        self.metrics = TrackLinkingMetrics()
        
        # Create checkpoint directory
        os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config"""
        optimizer_config = self.config['training']['optimizer']
        
        if optimizer_config['type'] == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config['weight_decay']
            )
        elif optimizer_config['type'] == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                momentum=optimizer_config['momentum'],
                weight_decay=optimizer_config['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config['type']}")
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        scheduler_config = self.config['training']['scheduler']
        
        if scheduler_config['type'] == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config['step_size'],
                gamma=scheduler_config['gamma']
            )
        elif scheduler_config['type'] == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config['T_max']
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_config['type']}")
    
    def _compute_combined_loss(self, 
                              outputs: Dict[str, torch.Tensor],
                              targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss from multiple components
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
        
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        losses = {}
        
        # Feature learning loss (triplet or contrastive)
        if 'anchor_features' in outputs and 'positive_features' in outputs and 'negative_features' in outputs:
            losses['triplet'] = self.triplet_loss(
                outputs['anchor_features'],
                outputs['positive_features'],
                outputs['negative_features']
            )
        
        # Tracklet association loss
        if 'tracklet_features_1' in outputs and 'tracklet_features_2' in outputs and 'association_labels' in targets:
            losses['contrastive'] = self.contrastive_loss(
                outputs['tracklet_features_1'],
                outputs['tracklet_features_2'],
                targets['association_labels']
            )
        
        # Motion prediction loss
        if 'predicted_motion' in outputs and 'target_motion' in targets:
            losses['motion'] = self.mse_loss(
                outputs['predicted_motion'],
                targets['target_motion']
            )
        
        # Combine losses with weights
        loss_weights = self.config['training']['loss_weights']
        total_loss = sum(loss_weights.get(name, 1.0) * loss for name, loss in losses.items())
        
        # Convert to float for logging
        loss_dict = {name: loss.item() for name, loss in losses.items()}
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            outputs = self.model(
                batch['tracklet_images'],
                batch['bbox_sequence']
            )
            
            # Compute loss
            total_loss, loss_dict = self._compute_combined_loss(outputs, batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Log losses
            epoch_losses.append(loss_dict)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Log to tensorboard
            if batch_idx % self.config['logging']['log_interval'] == 0:
                step = self.current_epoch * len(train_loader) + batch_idx
                for name, value in loss_dict.items():
                    self.writer.add_scalar(f'train/{name}_loss', value, step)
        
        # Compute average losses
        avg_losses = {}
        for key in epoch_losses[0].keys():
            avg_losses[key] = np.mean([loss[key] for loss in epoch_losses])
        
        return avg_losses
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        val_losses = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(
                    batch['tracklet_images'],
                    batch['bbox_sequence']
                )
                
                # Compute loss
                total_loss, loss_dict = self._compute_combined_loss(outputs, batch)
                val_losses.append(loss_dict)
                
                # Collect predictions for metrics
                if 'tracklet_features_1' in outputs and 'tracklet_features_2' in outputs:
                    # Compute similarity scores
                    feat1 = outputs['tracklet_features_1']
                    feat2 = outputs['tracklet_features_2']
                    similarities = torch.cosine_similarity(feat1, feat2, dim=1)
                    
                    all_predictions.extend(similarities.cpu().numpy())
                    all_targets.extend(batch['association_labels'].cpu().numpy())
        
        # Compute average validation losses
        avg_losses = {}
        for key in val_losses[0].keys():
            avg_losses[key] = np.mean([loss[key] for loss in val_losses])
        
        # Compute validation accuracy
        if all_predictions and all_targets:
            # Convert similarities to binary predictions (threshold = 0.5)
            predictions = np.array(all_predictions) > 0.5
            targets = np.array(all_targets)
            accuracy = np.mean(predictions == targets)
            avg_losses['accuracy'] = accuracy
        
        return avg_losses
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device"""
        device_batch = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_accuracy': self.best_accuracy,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config['training']['checkpoint_dir'],
            f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(
                self.config['training']['checkpoint_dir'],
                'best_model.pth'
            )
            torch.save(checkpoint, best_path)
            print(f"New best model saved with accuracy: {self.best_accuracy:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_accuracy = checkpoint['best_accuracy']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        """Main training loop"""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model has {sum(p.numel() for p in self.model.parameters())} parameters")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_losses = self.train_epoch(train_loader)
            self.train_losses.append(train_losses)
            
            # Validate epoch
            val_losses = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Check if best model
            current_accuracy = val_losses.get('accuracy', 0.0)
            is_best = current_accuracy > self.best_accuracy
            if is_best:
                self.best_accuracy = current_accuracy
            
            self.val_accuracies.append(current_accuracy)
            
            # Log to tensorboard
            for name, value in train_losses.items():
                self.writer.add_scalar(f'epoch/train_{name}', value, epoch)
            for name, value in val_losses.items():
                self.writer.add_scalar(f'epoch/val_{name}', value, epoch)
            
            # Print epoch summary
            print(f"Epoch {epoch + 1}/{num_epochs}:")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"  Val Loss: {val_losses['total']:.4f}")
            print(f"  Val Accuracy: {current_accuracy:.4f}")
            print(f"  Best Accuracy: {self.best_accuracy:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config['training']['save_interval'] == 0:
                self.save_checkpoint(epoch + 1, is_best)
        
        # Save final checkpoint
        self.save_checkpoint(num_epochs, False)
        
        # Plot training curves
        self.plot_training_curves()
        
        print("Training completed!")
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        epochs = range(1, len(self.train_losses) + 1)
        train_total_losses = [loss['total'] for loss in self.train_losses]
        
        axes[0].plot(epochs, train_total_losses, label='Train Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot accuracies
        axes[1].plot(epochs, self.val_accuracies, label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['logging']['output_dir'], 'training_curves.png'))
        plt.close()


def create_synthetic_data_loader(config: Dict, split: str = 'train') -> DataLoader:
    """
    Create synthetic data loader for demonstration
    In practice, this should load real face tracking data
    """
    from torch.utils.data import Dataset
    
    class SyntheticTrackletDataset(Dataset):
        def __init__(self, num_samples: int = 1000):
            self.num_samples = num_samples
            self.seq_len = 5
            self.img_size = (224, 224)
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Generate synthetic tracklet images
            tracklet_images = torch.randn(self.seq_len, 3, *self.img_size)
            
            # Generate synthetic bounding box sequence
            bbox_sequence = torch.randn(self.seq_len, 4)
            
            # Generate synthetic labels
            association_labels = torch.randint(0, 2, (1,)).float()
            
            return {
                'tracklet_images': tracklet_images,
                'bbox_sequence': bbox_sequence,
                'association_labels': association_labels,
                'target_motion': torch.randn(4)  # Next bounding box
            }
    
    dataset = SyntheticTrackletDataset(
        num_samples=config['data'][f'{split}_samples']
    )
    
    return DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(split == 'train'),
        num_workers=config['data']['num_workers']
    )


def load_config() -> Dict:
    """Load training configuration"""
    return {
        'model': {
            'backbone': 'resnet50',
            'embedding_dim': 512,
            'max_distance': 0.7
        },
        'training': {
            'batch_size': 16,
            'num_epochs': 50,
            'optimizer': {
                'type': 'adam',
                'lr': 0.001,
                'weight_decay': 1e-4
            },
            'scheduler': {
                'type': 'step',
                'step_size': 20,
                'gamma': 0.1
            },
            'loss_weights': {
                'triplet': 1.0,
                'contrastive': 1.0,
                'motion': 0.5
            },
            'triplet_margin': 0.3,
            'contrastive_margin': 1.0,
            'checkpoint_dir': 'checkpoints',
            'save_interval': 10
        },        'data': {
            'train_samples': 1000,
            'val_samples': 200,
            'num_workers': 0  # Set to 0 for Windows compatibility
        },
        'logging': {
            'tensorboard_dir': 'logs/face_track_linking',
            'output_dir': 'outputs',
            'log_interval': 50
        }
    }


def main():
    """Main training function"""
    # Load configuration
    config = load_config()
    
    # Create output directories
    os.makedirs(config['logging']['output_dir'], exist_ok=True)
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    
    # Save configuration
    with open(os.path.join(config['logging']['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = create_synthetic_data_loader(config, 'train')
    val_loader = create_synthetic_data_loader(config, 'val')
    
    # Create trainer
    print("Initializing trainer...")
    trainer = FaceTrackLinkingTrainer(config)
    
    # Start training
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs']
    )
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
