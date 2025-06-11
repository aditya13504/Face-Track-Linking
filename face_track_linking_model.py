"""
Face Track Linking Model based on Research Analysis

This module implements a comprehensive face track linking system based on
insights from the research papers in our curated dataset.

Key Research Insights Applied:
1. Bipartite matching with Hungarian algorithm (from FaceSORT paper)
2. Tracklet-to-tracklet association (from multiple papers)
3. Deep feature extraction with ResNet/VGG (from multiple papers)
4. Combined appearance and motion affinity (from tracklet association papers)
5. SVM-based linking as baseline (from Novel Method paper)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy.optimize import linear_sum_assignment
import cv2
from collections import defaultdict


class DeepFeatureExtractor(nn.Module):
    """
    Deep CNN feature extractor for face representation
    Based on research insights from FaceSORT and other papers
    """
    
    def __init__(self, 
                 backbone: str = 'resnet50',
                 embedding_dim: int = 512,
                 pretrained: bool = True):
        super(DeepFeatureExtractor, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Load backbone CNN
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'vgg16':
            self.backbone = models.vgg16(pretrained=pretrained)
            backbone_dim = self.backbone.classifier[6].in_features
            self.backbone.classifier[6] = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Feature projection layer
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim * 2),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
    def forward(self, x):
        # Extract backbone features
        features = self.backbone(x)
        
        # Project to embedding space
        embeddings = self.projection(features)
        
        # L2 normalize for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class MotionPredictor(nn.Module):
    """
    Motion prediction model using Kalman filter principles
    Based on research insights about motion affinity
    """
    
    def __init__(self, state_dim: int = 4):
        super(MotionPredictor, self).__init__()
        self.state_dim = state_dim  # [x, y, w, h] 
        
        # Motion model that takes [x, y, w, h, vx, vy, vw, vh] -> [x, y, w, h]
        # Input is bbox + velocity (8 dimensions), output is next bbox (4 dimensions)
        self.motion_model = nn.Sequential(
            nn.Linear(state_dim * 2, 64),  # Input: 8 dimensions (bbox + velocity)
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim)  # Output: 4 dimensions (next bbox)
        )
        
    def forward(self, bbox_sequence):
        """
        Predict next bounding box given sequence of previous boxes
        
        Args:
            bbox_sequence: [batch_size, seq_len, 4] (x, y, w, h)
        Returns:
            predicted_bbox: [batch_size, 4]
        """
        batch_size, seq_len, _ = bbox_sequence.shape
        
        # Simple velocity calculation
        if seq_len >= 2:
            velocity = bbox_sequence[:, -1] - bbox_sequence[:, -2]
            current_state = torch.cat([bbox_sequence[:, -1], velocity], dim=1)
        else:
            current_state = torch.cat([bbox_sequence[:, -1], 
                                     torch.zeros_like(bbox_sequence[:, -1])], dim=1)
        
        # Predict next state
        next_state = self.motion_model(current_state)
        predicted_bbox = next_state[:, :4]  # Extract position
        
        return predicted_bbox


class AffinityNetwork(nn.Module):
    """
    Computes appearance and motion affinity between tracklets
    Based on research insights from tracklet association papers
    """
    
    def __init__(self, embedding_dim: int = 512):
        super(AffinityNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Appearance affinity network
        self.appearance_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Motion affinity network
        self.motion_net = nn.Sequential(
            nn.Linear(8, 64),  # Two bounding boxes
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Fusion weights
        self.appearance_weight = nn.Parameter(torch.tensor(0.7))
        self.motion_weight = nn.Parameter(torch.tensor(0.3))
        
    def forward(self, 
                features1: torch.Tensor,
                features2: torch.Tensor,
                bbox1: torch.Tensor,
                bbox2: torch.Tensor) -> torch.Tensor:
        """
        Compute affinity between two tracklets
        
        Args:
            features1: [batch_size, embedding_dim] - first tracklet features
            features2: [batch_size, embedding_dim] - second tracklet features
            bbox1: [batch_size, 4] - first tracklet bbox
            bbox2: [batch_size, 4] - second tracklet bbox
        
        Returns:
            affinity: [batch_size, 1] - combined affinity score
        """
        # Appearance affinity
        appearance_input = torch.cat([features1, features2], dim=1)
        appearance_affinity = self.appearance_net(appearance_input)
        
        # Motion affinity (based on IoU and distance)
        motion_input = torch.cat([bbox1, bbox2], dim=1)
        motion_affinity = self.motion_net(motion_input)
        
        # Weighted combination
        combined_affinity = (self.appearance_weight * appearance_affinity + 
                           self.motion_weight * motion_affinity)
        
        return combined_affinity


class TrackletAssociation(nn.Module):
    """
    Tracklet-to-tracklet association module
    Implements Hungarian algorithm for optimal assignment
    """
    
    def __init__(self, max_distance: float = 0.7):
        super(TrackletAssociation, self).__init__()
        self.max_distance = max_distance
    
    def compute_cost_matrix(self, 
                          tracklets1: List[Dict],
                          tracklets2: List[Dict],
                          affinity_net: AffinityNetwork) -> np.ndarray:
        """
        Compute cost matrix between two sets of tracklets
        
        Args:
            tracklets1: List of tracklet dictionaries
            tracklets2: List of tracklet dictionaries
            affinity_net: Trained affinity network
        
        Returns:
            cost_matrix: [len(tracklets1), len(tracklets2)]
        """
        n1, n2 = len(tracklets1), len(tracklets2)
        cost_matrix = np.zeros((n1, n2))
        
        for i, t1 in enumerate(tracklets1):
            for j, t2 in enumerate(tracklets2):
                # Extract features and bounding boxes
                feat1 = t1['features']  # Averaged features
                feat2 = t2['features']
                bbox1 = t1['bbox']     # Last bounding box
                bbox2 = t2['bbox']
                
                # Convert to tensors
                feat1_tensor = torch.tensor(feat1).unsqueeze(0).float()
                feat2_tensor = torch.tensor(feat2).unsqueeze(0).float()
                bbox1_tensor = torch.tensor(bbox1).unsqueeze(0).float()
                bbox2_tensor = torch.tensor(bbox2).unsqueeze(0).float()
                
                # Compute affinity
                with torch.no_grad():
                    affinity = affinity_net(feat1_tensor, feat2_tensor, 
                                          bbox1_tensor, bbox2_tensor)
                
                # Convert affinity to cost (1 - affinity)
                cost_matrix[i, j] = 1.0 - affinity.item()
        
        return cost_matrix
    
    def hungarian_assignment(self, cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """
        Perform Hungarian algorithm for optimal assignment
        
        Args:
            cost_matrix: [n1, n2] cost matrix
        
        Returns:
            assignments: List of (i, j) pairs
        """
        # Use scipy's implementation of Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        assignments = []
        for i, j in zip(row_indices, col_indices):
            if cost_matrix[i, j] < self.max_distance:
                assignments.append((i, j))
        
        return assignments


class SVMLinking(nn.Module):
    """
    SVM-based linking as baseline method
    Based on "A Novel Method for Face Track Linking in Videos"
    """
    
    def __init__(self, embedding_dim: int = 512):
        super(SVMLinking, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Simple linear classifier as SVM approximation
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """
        Predict if two face features belong to the same person
        
        Args:
            features1: [batch_size, embedding_dim]
            features2: [batch_size, embedding_dim]
        
        Returns:
            similarity: [batch_size, 1] - probability of same identity
        """
        combined_features = torch.cat([features1, features2], dim=1)
        similarity = self.classifier(combined_features)
        return similarity


class FaceTrackLinkingModel(nn.Module):
    """
    Complete Face Track Linking Model
    Integrates all components based on research insights
    """
    
    def __init__(self, 
                 backbone: str = 'resnet50',
                 embedding_dim: int = 512,
                 max_distance: float = 0.7):
        super(FaceTrackLinkingModel, self).__init__()
        
        # Core components
        self.feature_extractor = DeepFeatureExtractor(backbone, embedding_dim)
        self.motion_predictor = MotionPredictor()
        self.affinity_network = AffinityNetwork(embedding_dim)
        self.tracklet_association = TrackletAssociation(max_distance)
        self.svm_linking = SVMLinking(embedding_dim)
        
        self.embedding_dim = embedding_dim
        
    def extract_tracklet_features(self, tracklet_images: torch.Tensor) -> torch.Tensor:
        """
        Extract aggregated features for a tracklet
        
        Args:
            tracklet_images: [seq_len, 3, H, W] - images in tracklet
        
        Returns:
            aggregated_features: [embedding_dim] - tracklet representation
        """
        seq_len = tracklet_images.shape[0]
        
        # Extract features for all images
        features = self.feature_extractor(tracklet_images)  # [seq_len, embedding_dim]
        
        # Aggregate using average pooling (can be improved with attention)
        aggregated_features = torch.mean(features, dim=0)
        
        # Ensure normalized
        aggregated_features = F.normalize(aggregated_features, p=2, dim=0)
        
        return aggregated_features
    
    def predict_motion(self, bbox_sequence: torch.Tensor) -> torch.Tensor:
        """
        Predict next bounding box location
        
        Args:
            bbox_sequence: [seq_len, 4] - sequence of bounding boxes
        
        Returns:
            predicted_bbox: [4] - predicted next bounding box
        """
        bbox_sequence = bbox_sequence.unsqueeze(0)  # Add batch dimension
        predicted_bbox = self.motion_predictor(bbox_sequence)
        return predicted_bbox.squeeze(0)
    
    def compute_tracklet_affinity(self, 
                                tracklet1: Dict,
                                tracklet2: Dict) -> float:
        """
        Compute affinity between two tracklets
        
        Args:
            tracklet1: Dictionary with 'features' and 'bbox'
            tracklet2: Dictionary with 'features' and 'bbox'
        
        Returns:
            affinity: Similarity score between tracklets
        """
        # Convert to tensors
        feat1 = torch.tensor(tracklet1['features']).unsqueeze(0).float()
        feat2 = torch.tensor(tracklet2['features']).unsqueeze(0).float()
        bbox1 = torch.tensor(tracklet1['bbox']).unsqueeze(0).float()
        bbox2 = torch.tensor(tracklet2['bbox']).unsqueeze(0).float()
        
        # Compute affinity
        with torch.no_grad():
            affinity = self.affinity_network(feat1, feat2, bbox1, bbox2)
        
        return affinity.item()
    
    def link_tracklets(self, 
                      tracklets1: List[Dict],
                      tracklets2: List[Dict]) -> List[Tuple[int, int]]:
        """
        Link tracklets between two time windows using Hungarian algorithm
        
        Args:
            tracklets1: List of tracklets from first time window
            tracklets2: List of tracklets from second time window
        
        Returns:
            assignments: List of (i, j) pairs indicating linked tracklets
        """
        return self.tracklet_association.hungarian_assignment(
            self.tracklet_association.compute_cost_matrix(
                tracklets1, tracklets2, self.affinity_network
            )
        )
    
    def forward(self, 
                crop1: torch.Tensor,
                crop2: torch.Tensor = None,
                temporal_gap: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass that handles both individual crops and tracklet sequences
        
        Args:
            crop1: [batch_size, 3, H, W] - First face crop OR 
                   [batch_size, seq_len, 3, H, W] - Tracklet image sequence
            crop2: [batch_size, 3, H, W] - Second face crop (optional for sequences)
            temporal_gap: [batch_size, 1] - Temporal gap between crops (optional)
        
        Returns:
            outputs: Dictionary with similarity predictions and features
        """
        # Check if we have individual crops or tracklet sequence
        if crop2 is not None:
            # Individual crops mode: [batch_size, 3, H, W]
            features1 = self.feature_extractor(crop1)
            features2 = self.feature_extractor(crop2)
        else:
            # Tracklet sequence mode: [batch_size, seq_len, 3, H, W]
            if len(crop1.shape) == 5:
                batch_size, seq_len, c, h, w = crop1.shape
                
                # Reshape to process all frames at once: [batch_size * seq_len, 3, H, W]
                tracklet_flat = crop1.view(batch_size * seq_len, c, h, w)
                
                # Extract features from all frames
                frame_features = self.feature_extractor(tracklet_flat)  # [batch_size * seq_len, feature_dim]
                
                # Reshape back to sequence: [batch_size, seq_len, feature_dim]
                features = frame_features.view(batch_size, seq_len, -1)
                
                # Use first and last frame features for similarity computation
                features1 = features[:, 0, :]  # First frame
                features2 = features[:, -1, :] # Last frame
            else:
                # Single crop passed as crop1, treat as features1 = features2
                features1 = self.feature_extractor(crop1)
                features2 = features1
        
        # Compute similarity score using SVM linking
        similarity_score = self.svm_linking(features1, features2)
        
        return {
            'similarity_score': similarity_score,
            'features1': features1,
            'features2': features2
        }


def create_face_track_linking_model(config: Dict) -> FaceTrackLinkingModel:
    """
    Factory function to create face track linking model
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        model: Initialized FaceTrackLinkingModel
    """
    return FaceTrackLinkingModel(
        backbone=config.get('backbone', 'resnet50'),
        embedding_dim=config.get('embedding_dim', 512),
        max_distance=config.get('max_distance', 0.7)
    )


# Research-based loss functions
class TripletLoss(nn.Module):
    """
    Triplet loss for learning discriminative features
    """
    
    def __init__(self, margin: float = 0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for tracklet association
    """
    
    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, features1, features2, labels):
        """
        Args:
            features1, features2: [batch_size, embedding_dim]
            labels: [batch_size] - 1 if same identity, 0 if different
        """
        distances = F.pairwise_distance(features1, features2, p=2)
        
        # Loss for positive pairs (same identity)
        pos_loss = labels * torch.pow(distances, 2)
        
        # Loss for negative pairs (different identity)
        neg_loss = (1 - labels) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)
        
        loss = (pos_loss + neg_loss).mean()
        return loss


# Evaluation metrics based on research insights
class TrackLinkingMetrics:
    """
    Evaluation metrics for face track linking
    Based on metrics found in research papers
    """
    
    @staticmethod
    def track_purity(predicted_links: List[Tuple], ground_truth: List[Tuple]) -> float:
        """Calculate track purity metric"""
        if not predicted_links:
            return 0.0
        
        correct_links = set(predicted_links) & set(ground_truth)
        return len(correct_links) / len(predicted_links)
    
    @staticmethod
    def cluster_fragmentation(tracklets_per_identity: Dict[int, List[int]]) -> float:
        """Calculate cluster fragmentation metric"""
        total_fragmentation = 0
        total_identities = len(tracklets_per_identity)
        
        for identity, tracklets in tracklets_per_identity.items():
            fragmentation = len(tracklets)  # Number of tracklets for this identity
            total_fragmentation += fragmentation
        
        return total_fragmentation / total_identities if total_identities > 0 else 0.0
    
    @staticmethod
    def linking_accuracy(predicted_links: List[Tuple], ground_truth: List[Tuple]) -> Dict[str, float]:
        """Calculate comprehensive linking accuracy metrics"""
        predicted_set = set(predicted_links)
        ground_truth_set = set(ground_truth)
        
        true_positives = len(predicted_set & ground_truth_set)
        false_positives = len(predicted_set - ground_truth_set)
        false_negatives = len(ground_truth_set - predicted_set)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
