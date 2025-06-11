"""
Data Loader for Face Track Linking Training

This module provides data loading functionality for training the face track linking model
with synthetic or real annotated data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import torchvision.transforms as transforms
import random
from collections import defaultdict


class FaceTrackLinkingDataset(Dataset):
    """
    Dataset class for face track linking training
    """
    
    def __init__(self, 
                 data_dir: str,
                 pairs_file: str = "training_pairs.json",
                 transform: Optional[transforms.Compose] = None,
                 mode: str = 'train',
                 max_temporal_gap: int = 50):
        """
        Initialize dataset
        
        Args:
            data_dir: Directory containing the dataset
            pairs_file: JSON file with training pairs
            transform: Data augmentation transforms
            mode: 'train', 'val', or 'test'
            max_temporal_gap: Maximum temporal gap between pairs
        """
        self.data_dir = Path(data_dir)
        self.crops_dir = self.data_dir / "crops"
        self.mode = mode
        self.max_temporal_gap = max_temporal_gap
        
        # Load training pairs
        pairs_path = self.data_dir / pairs_file
        with open(pairs_path, 'r') as f:
            self.pairs = json.load(f)
        
        # Filter pairs by temporal gap
        self.pairs = [p for p in self.pairs if p['temporal_gap'] <= max_temporal_gap]
        
        # Split data
        self._split_data()
        
        # Setup transforms
        if transform is None:
            self.transform = self._default_transforms()
        else:
            self.transform = transform
        
        print(f"Loaded {len(self.current_pairs)} pairs for {mode} mode")
    
    def _split_data(self):
        """Split data into train/val/test"""
        # Shuffle with fixed seed for reproducibility
        random.seed(42)
        shuffled_pairs = self.pairs.copy()
        random.shuffle(shuffled_pairs)
        
        n_pairs = len(shuffled_pairs)
        
        if self.mode == 'train':
            self.current_pairs = shuffled_pairs[:int(0.7 * n_pairs)]
        elif self.mode == 'val':
            self.current_pairs = shuffled_pairs[int(0.7 * n_pairs):int(0.85 * n_pairs)]
        else:  # test
            self.current_pairs = shuffled_pairs[int(0.85 * n_pairs):]
    
    def _default_transforms(self):
        """Default transforms for face crops"""
        if self.mode == 'train':
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((112, 112)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.current_pairs)
    
    def __getitem__(self, idx):
        pair = self.current_pairs[idx]
        
        # Load crops
        crop1_path = self.crops_dir / pair['crop1_path']
        crop2_path = self.crops_dir / pair['crop2_path']
        
        crop1 = cv2.imread(str(crop1_path))
        crop2 = cv2.imread(str(crop2_path))
        
        if crop1 is None or crop2 is None:
            # Handle missing files by creating dummy data
            crop1 = np.zeros((112, 112, 3), dtype=np.uint8)
            crop2 = np.zeros((112, 112, 3), dtype=np.uint8)
        
        # Convert BGR to RGB
        crop1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2RGB)
        crop2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        crop1 = self.transform(crop1)
        crop2 = self.transform(crop2)
        
        # Prepare other features
        temporal_gap = pair['temporal_gap']
        label = pair['label']
        
        # Normalize temporal gap
        temporal_gap_norm = min(temporal_gap / self.max_temporal_gap, 1.0)
        
        return {
            'crop1': crop1,
            'crop2': crop2,
            'temporal_gap': torch.tensor(temporal_gap_norm, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long),
            'metadata': {
                'track1_id': pair['track1_id'],
                'track2_id': pair['track2_id'],
                'frame1_id': pair['frame1_id'],
                'frame2_id': pair['frame2_id'],
                'identity_id': pair.get('identity_id', -1)
            }
        }


class TrackletDataset(Dataset):
    """
    Dataset for loading tracklets (sequences of detections from same track)
    """
    
    def __init__(self,
                 data_dir: str,
                 metadata_file: str = "dataset_metadata.json",
                 min_tracklet_length: int = 5,
                 max_tracklet_length: int = 20,
                 transform: Optional[transforms.Compose] = None):
        """
        Initialize tracklet dataset
        
        Args:
            data_dir: Directory containing the dataset
            metadata_file: JSON file with dataset metadata
            min_tracklet_length: Minimum tracklet length
            max_tracklet_length: Maximum tracklet length
            transform: Data augmentation transforms
        """
        self.data_dir = Path(data_dir)
        self.crops_dir = self.data_dir / "crops"
        self.min_tracklet_length = min_tracklet_length
        self.max_tracklet_length = max_tracklet_length
        
        # Load metadata
        metadata_path = self.data_dir / metadata_file
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Extract tracklets
        self.tracklets = self._extract_tracklets()
        
        # Setup transforms
        if transform is None:
            self.transform = self._default_transforms()
        else:
            self.transform = transform
        
        print(f"Loaded {len(self.tracklets)} tracklets")
    
    def _extract_tracklets(self) -> List[Dict]:
        """Extract tracklets from metadata"""
        tracklets = []
        
        for video_data in self.metadata['videos']:
            video_id = video_data['video_id']
            tracks = video_data['tracks']
            frames = video_data['frames']
            
            # Group detections by track
            track_detections = defaultdict(list)
            
            for frame_data in frames:
                for det in frame_data['detections']:
                    track_detections[det['track_id']].append({
                        'frame_id': frame_data['frame_id'],
                        'bbox': det['bbox'],
                        'center': det['center'],
                        'crop_path': (f"video_{video_id:04d}_frame_"
                                    f"{frame_data['frame_id']:04d}_track_{det['track_id']:04d}.jpg")
                    })
            
            # Create tracklets
            for track_id, detections in track_detections.items():
                if len(detections) >= self.min_tracklet_length:
                    # Sort by frame
                    detections.sort(key=lambda x: x['frame_id'])
                    
                    # Split long tracklets into segments
                    for start_idx in range(0, len(detections), self.max_tracklet_length):
                        end_idx = min(start_idx + self.max_tracklet_length, len(detections))
                        segment = detections[start_idx:end_idx]
                        
                        if len(segment) >= self.min_tracklet_length:
                            tracklet = {
                                'video_id': video_id,
                                'track_id': track_id,
                                'identity_id': tracks[track_id]['identity_id'],
                                'detections': segment,
                                'length': len(segment)
                            }
                            tracklets.append(tracklet)
        
        return tracklets
    
    def _default_transforms(self):
        """Default transforms for tracklet crops"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.tracklets)
    
    def __getitem__(self, idx):
        tracklet = self.tracklets[idx]
        
        # Load all crops in tracklet
        crops = []
        positions = []
        frame_ids = []
        
        for det in tracklet['detections']:
            crop_path = self.crops_dir / det['crop_path']
            crop = cv2.imread(str(crop_path))
            
            if crop is None:
                crop = np.zeros((112, 112, 3), dtype=np.uint8)
            
            # Convert BGR to RGB
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = self.transform(crop)
            crops.append(crop)
            
            # Normalized position
            pos = [det['center'][0] / 640.0, det['center'][1] / 480.0]  # Assuming 640x480 frames
            positions.append(pos)
            
            frame_ids.append(det['frame_id'])
        
        # Pad or truncate to max length
        target_length = self.max_tracklet_length
        
        if len(crops) < target_length:
            # Pad with last frame
            last_crop = crops[-1]
            last_pos = positions[-1]
            last_frame = frame_ids[-1]
            
            while len(crops) < target_length:
                crops.append(last_crop.clone())
                positions.append(last_pos.copy())
                frame_ids.append(last_frame)
        
        else:
            # Truncate
            crops = crops[:target_length]
            positions = positions[:target_length]
            frame_ids = frame_ids[:target_length]
        
        # Stack crops
        crops_tensor = torch.stack(crops)  # [T, C, H, W]
        positions_tensor = torch.tensor(positions, dtype=torch.float32)  # [T, 2]
        frame_ids_tensor = torch.tensor(frame_ids, dtype=torch.long)  # [T]
        
        # Create mask for valid frames
        mask = torch.zeros(target_length, dtype=torch.bool)
        mask[:tracklet['length']] = True
        
        return {
            'crops': crops_tensor,
            'positions': positions_tensor,
            'frame_ids': frame_ids_tensor,
            'mask': mask,
            'identity_id': tracklet['identity_id'],
            'track_id': tracklet['track_id'],
            'video_id': tracklet['video_id'],
            'length': tracklet['length']
        }


def create_data_loaders(data_dir: str,
                       batch_size: int = 32,
                       num_workers: int = 4,
                       dataset_type: str = 'pairs') -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders
    
    Args:
        data_dir: Directory containing the dataset
        batch_size: Batch size for training
        num_workers: Number of worker processes
        dataset_type: 'pairs' or 'tracklets'
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    if dataset_type == 'pairs':
        # Create pair datasets
        train_dataset = FaceTrackLinkingDataset(data_dir, mode='train')
        val_dataset = FaceTrackLinkingDataset(data_dir, mode='val')
        test_dataset = FaceTrackLinkingDataset(data_dir, mode='test')
        
    elif dataset_type == 'tracklets':
        # Create tracklet dataset and split
        full_dataset = TrackletDataset(data_dir)
        
        # Split tracklets
        n_tracklets = len(full_dataset)
        train_size = int(0.7 * n_tracklets)
        val_size = int(0.15 * n_tracklets)
        test_size = n_tracklets - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
    
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader


def collate_tracklets(batch):
    """Custom collate function for variable-length tracklets"""
    # This would be used if we need variable-length tracklets
    # For now, we pad to fixed length in __getitem__
    return torch.utils.data.default_collate(batch)


if __name__ == '__main__':
    # Test data loading
    data_dir = "synthetic_data"
    
    if Path(data_dir).exists():
        print("Testing pair dataset...")
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir, batch_size=4, dataset_type='pairs'
        )
        
        # Test one batch
        for batch in train_loader:
            print(f"Batch shape: {batch['crop1'].shape}")
            print(f"Labels: {batch['label']}")
            print(f"Temporal gaps: {batch['temporal_gap']}")
            break
        
        print("\nTesting tracklet dataset...")
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir, batch_size=2, dataset_type='tracklets'
        )
        
        # Test one batch
        for batch in train_loader:
            print(f"Crops shape: {batch['crops'].shape}")
            print(f"Positions shape: {batch['positions'].shape}")
            print(f"Identity IDs: {batch['identity_id']}")
            break
    
    else:
        print(f"Dataset directory {data_dir} not found. Run synthetic_data_generator.py first.")
