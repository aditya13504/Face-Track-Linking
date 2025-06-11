"""
Synthetic Data Generator for Face Track Linking

This module generates synthetic training data for the face track linking model
when real annotated video data is not available.

It creates tracklets with known ground truth associations to train the model.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import random
from collections import defaultdict
import face_recognition
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


class SyntheticFaceTrackGenerator:
    """
    Generates synthetic face tracking data for training
    """
    
    def __init__(self, 
                 output_dir: str = "synthetic_data",
                 num_identities: int = 10,
                 num_videos: int = 100,
                 frames_per_video: int = 50,
                 max_faces_per_frame: int = 3,
                 image_size: Tuple[int, int] = (640,480),
                 face_size: Tuple[int, int] = (112, 112)):
        """
        Initialize synthetic data generator
        
        Args:
            output_dir: Directory to save generated data
            num_identities: Number of unique identities to generate
            num_videos: Number of synthetic videos to create
            frames_per_video: Frames per video sequence
            max_faces_per_frame: Maximum faces per frame
            image_size: Size of generated frames
            face_size: Size of face crops
        """
        self.output_dir = Path(output_dir)
        self.num_identities = num_identities
        self.num_videos = num_videos
        self.frames_per_video = frames_per_video
        self.max_faces_per_frame = max_faces_per_frame
        self.image_size = image_size
        self.face_size = face_size
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "frames").mkdir(exist_ok=True)
        (self.output_dir / "crops").mkdir(exist_ok=True)
        (self.output_dir / "annotations").mkdir(exist_ok=True)
        
        # Generate base identity templates
        self.identity_templates = self._generate_identity_templates()
    
    def _generate_identity_templates(self) -> List[np.ndarray]:
        """
        Generate base templates for each identity
        
        Returns:
            List of identity templates
        """
        templates = []
        
        for i in range(self.num_identities):
            # Create a simple geometric template for each identity
            template = np.random.randint(50, 200, self.face_size + (3,), dtype=np.uint8)
            
            # Add some distinctive features
            # Random colored shapes to make identities distinguishable
            overlay = Image.fromarray(template)
            draw = ImageDraw.Draw(overlay)
            
            # Random geometric patterns for each identity
            np.random.seed(i)  # Consistent per identity
            
            # Draw some shapes
            for _ in range(3):
                x1, y1 = np.random.randint(0, self.face_size[0]//2, 2)
                x2, y2 = x1 + np.random.randint(10, 30), y1 + np.random.randint(10, 30)
                color = tuple(np.random.randint(0, 255, 3))
                draw.rectangle([x1, y1, x2, y2], fill=color)
            
            # Draw some circles
            for _ in range(2):
                center = np.random.randint(10, self.face_size[0]-10, 2)
                radius = np.random.randint(5, 15)
                color = tuple(np.random.randint(0, 255, 3))
                draw.ellipse([center[0]-radius, center[1]-radius, 
                             center[0]+radius, center[1]+radius], fill=color)
            
            templates.append(np.array(overlay))
        
        return templates
    
    def _generate_variations(self, template: np.ndarray, num_variations: int = 10) -> List[np.ndarray]:
        """
        Generate variations of an identity template
        
        Args:
            template: Base template
            num_variations: Number of variations to generate
            
        Returns:
            List of template variations
        """
        variations = []
        
        for _ in range(num_variations):
            variation = template.copy()
            
            # Add noise
            noise = np.random.normal(0, 10, template.shape).astype(np.int16)
            variation = np.clip(variation.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Random brightness/contrast
            alpha = np.random.uniform(0.8, 1.2)  # Contrast
            beta = np.random.uniform(-20, 20)    # Brightness
            variation = np.clip(alpha * variation + beta, 0, 255).astype(np.uint8)
            
            # Random rotation
            angle = np.random.uniform(-15, 15)
            center = (self.face_size[0]//2, self.face_size[1]//2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            variation = cv2.warpAffine(variation, M, self.face_size)
            
            # Random scaling
            scale = np.random.uniform(0.9, 1.1)
            new_size = (int(self.face_size[0] * scale), int(self.face_size[1] * scale))
            variation = cv2.resize(variation, new_size)
            
            # Pad or crop back to original size
            if scale > 1.0:
                # Crop
                start_x = (new_size[0] - self.face_size[0]) // 2
                start_y = (new_size[1] - self.face_size[1]) // 2
                variation = variation[start_y:start_y+self.face_size[1], 
                                   start_x:start_x+self.face_size[0]]
            else:
                # Pad
                pad_x = (self.face_size[0] - new_size[0]) // 2
                pad_y = (self.face_size[1] - new_size[1]) // 2
                variation = cv2.copyMakeBorder(variation, pad_y, pad_y, pad_x, pad_x, 
                                             cv2.BORDER_CONSTANT, value=0)
                variation = cv2.resize(variation, self.face_size)  # Ensure exact size
            
            variations.append(variation)
        
        return variations
    
    def _generate_trajectory(self, start_pos: Tuple[int, int], 
                           num_frames: int, 
                           frame_size: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Generate smooth trajectory for a face across frames
        
        Args:
            start_pos: Starting position
            num_frames: Number of frames
            frame_size: Size of frame
            
        Returns:
            List of positions
        """
        trajectory = [start_pos]
        current_pos = list(start_pos)
        
        # Random walk with momentum
        velocity = [np.random.uniform(-2, 2), np.random.uniform(-2, 2)]
        
        for _ in range(num_frames - 1):
            # Update velocity with some randomness
            velocity[0] += np.random.uniform(-0.5, 0.5)
            velocity[1] += np.random.uniform(-0.5, 0.5)
            
            # Limit velocity
            velocity[0] = np.clip(velocity[0], -5, 5)
            velocity[1] = np.clip(velocity[1], -5, 5)
            
            # Update position
            current_pos[0] += velocity[0]
            current_pos[1] += velocity[1]
            
            # Bounce off walls
            if current_pos[0] < self.face_size[0]//2 or current_pos[0] > frame_size[0] - self.face_size[0]//2:
                velocity[0] *= -0.8
                current_pos[0] = np.clip(current_pos[0], self.face_size[0]//2, 
                                       frame_size[0] - self.face_size[0]//2)
            
            if current_pos[1] < self.face_size[1]//2 or current_pos[1] > frame_size[1] - self.face_size[1]//2:
                velocity[1] *= -0.8
                current_pos[1] = np.clip(current_pos[1], self.face_size[1]//2, 
                                       frame_size[1] - self.face_size[1]//2)
            
            trajectory.append((int(current_pos[0]), int(current_pos[1])))
        
        return trajectory
    
    def generate_video_sequence(self, video_id: int) -> Dict:
        """
        Generate a synthetic video sequence with face tracks
        
        Args:
            video_id: Unique video identifier
            
        Returns:
            Dictionary containing video metadata and annotations
        """
        video_data = {
            'video_id': video_id,
            'num_frames': self.frames_per_video,
            'image_size': self.image_size,
            'tracks': {},
            'frames': []
        }
        
        # Determine which identities appear in this video
        num_identities_in_video = np.random.randint(1, min(self.max_faces_per_frame + 1, 
                                                          self.num_identities + 1))
        identity_ids = np.random.choice(self.num_identities, num_identities_in_video, replace=False)
        
        # Generate variations for each identity
        identity_variations = {}
        for identity_id in identity_ids:
            identity_variations[identity_id] = self._generate_variations(
                self.identity_templates[identity_id], self.frames_per_video
            )
        
        # Plan tracks for each identity
        tracks = {}
        track_id = 0
        
        for identity_id in identity_ids:
            # Each identity might have multiple tracklets (tracks)
            num_tracklets = np.random.randint(1, 3)  # 1-2 tracklets per identity
            
            for _ in range(num_tracklets):
                # Random start and end frames for this tracklet
                start_frame = np.random.randint(0, self.frames_per_video // 2)
                tracklet_length = np.random.randint(10, self.frames_per_video - start_frame)
                end_frame = min(start_frame + tracklet_length, self.frames_per_video - 1)
                
                # Generate trajectory
                start_pos = (np.random.randint(self.face_size[0]//2, 
                                             self.image_size[0] - self.face_size[0]//2),
                           np.random.randint(self.face_size[1]//2, 
                                           self.image_size[1] - self.face_size[1]//2))
                
                trajectory = self._generate_trajectory(start_pos, end_frame - start_frame + 1, 
                                                     self.image_size)
                
                tracks[track_id] = {
                    'identity_id': identity_id,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'trajectory': trajectory
                }
                
                track_id += 1
        
        video_data['tracks'] = tracks
        
        # Generate frames
        for frame_idx in range(self.frames_per_video):
            frame = np.zeros(self.image_size[::-1] + (3,), dtype=np.uint8)  # Black background
            
            # Add some random background noise
            frame += np.random.randint(0, 30, frame.shape, dtype=np.uint8)
            
            frame_annotations = {
                'frame_id': frame_idx,
                'detections': []
            }
            
            # Place faces for active tracks
            for track_id, track_info in tracks.items():
                if track_info['start_frame'] <= frame_idx <= track_info['end_frame']:
                    trajectory_idx = frame_idx - track_info['start_frame']
                    pos = track_info['trajectory'][trajectory_idx]
                    identity_id = track_info['identity_id']
                    
                    # Get face variation for this frame
                    variation_idx = frame_idx % len(identity_variations[identity_id])
                    face_crop = identity_variations[identity_id][variation_idx]
                    
                    # Place face on frame
                    x, y = pos
                    x1 = x - self.face_size[0] // 2
                    y1 = y - self.face_size[1] // 2
                    x2 = x1 + self.face_size[0]
                    y2 = y1 + self.face_size[1]
                    
                    # Ensure bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(self.image_size[0], x2)
                    y2 = min(self.image_size[1], y2)
                    
                    if x2 > x1 and y2 > y1:
                        # Resize face crop if needed
                        crop_h, crop_w = y2 - y1, x2 - x1
                        if crop_h != self.face_size[1] or crop_w != self.face_size[0]:
                            face_crop = cv2.resize(face_crop, (crop_w, crop_h))
                        
                        # Alpha blending for more realistic placement
                        alpha = 0.9
                        frame[y1:y2, x1:x2] = (alpha * face_crop + 
                                             (1 - alpha) * frame[y1:y2, x1:x2]).astype(np.uint8)
                        
                        # Add detection annotation
                        detection = {
                            'track_id': track_id,
                            'identity_id': identity_id,
                            'bbox': [x1, y1, x2 - x1, y2 - y1],  # x, y, w, h
                            'center': [x, y],
                            'confidence': 1.0
                        }
                        frame_annotations['detections'].append(detection)
            
            # Save frame
            frame_path = self.output_dir / "frames" / f"video_{video_id:04d}_frame_{frame_idx:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            
            # Save frame crops
            for det in frame_annotations['detections']:
                x, y, w, h = det['bbox']
                crop = frame[y:y+h, x:x+w]
                crop_path = (self.output_dir / "crops" / 
                           f"video_{video_id:04d}_frame_{frame_idx:04d}_track_{det['track_id']:04d}.jpg")
                cv2.imwrite(str(crop_path), crop)
            
            video_data['frames'].append(frame_annotations)
        
        return video_data
    
    def generate_dataset(self) -> Dict:
        """
        Generate complete synthetic dataset
        
        Returns:
            Dataset metadata
        """
        print(f"Generating synthetic dataset with {self.num_videos} videos...")
        
        dataset_metadata = {
            'num_videos': self.num_videos,
            'num_identities': self.num_identities,
            'frames_per_video': self.frames_per_video,
            'image_size': self.image_size,
            'face_size': self.face_size,
            'videos': []
        }
        
        for video_id in range(self.num_videos):
            if video_id % 10 == 0:
                print(f"Generating video {video_id}/{self.num_videos}")
            
            video_data = self.generate_video_sequence(video_id)
            dataset_metadata['videos'].append(video_data)
            
            # Save individual video annotations
            annotation_path = self.output_dir / "annotations" / f"video_{video_id:04d}.json"
            with open(annotation_path, 'w') as f:
                json.dump(video_data, f, indent=2)
        
        # Save dataset metadata
        metadata_path = self.output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        print(f"Dataset generation complete! Saved to {self.output_dir}")
        return dataset_metadata
    
    def create_training_pairs(self, dataset_metadata: Dict) -> List[Dict]:
        """
        Create training pairs for the linking model
        
        Args:
            dataset_metadata: Generated dataset metadata
            
        Returns:
            List of training pairs with labels
        """
        training_pairs = []
        
        for video_data in dataset_metadata['videos']:
            tracks = video_data['tracks']
            frames = video_data['frames']
            
            # Group detections by identity for positive pairs
            identity_detections = defaultdict(list)
            
            for frame_data in frames:
                for det in frame_data['detections']:
                    identity_detections[det['identity_id']].append({
                        'frame_id': frame_data['frame_id'],
                        'track_id': det['track_id'],
                        'bbox': det['bbox'],
                        'center': det['center'],
                        'crop_path': (f"video_{video_data['video_id']:04d}_frame_"
                                    f"{frame_data['frame_id']:04d}_track_{det['track_id']:04d}.jpg")
                    })
            
            # Generate positive pairs (same identity)
            for identity_id, detections in identity_detections.items():
                for i in range(len(detections)):
                    for j in range(i + 1, min(i + 10, len(detections))):  # Limit pairs per identity
                        det1, det2 = detections[i], detections[j]
                        
                        # Skip pairs from same tracklet that are too close
                        if (det1['track_id'] == det2['track_id'] and 
                            abs(det1['frame_id'] - det2['frame_id']) < 3):
                            continue
                        
                        pair = {
                            'crop1_path': det1['crop_path'],
                            'crop2_path': det2['crop_path'],
                            'label': 1,  # Same identity
                            'identity_id': identity_id,
                            'track1_id': det1['track_id'],
                            'track2_id': det2['track_id'],
                            'frame1_id': det1['frame_id'],
                            'frame2_id': det2['frame_id'],
                            'temporal_gap': abs(det1['frame_id'] - det2['frame_id'])
                        }
                        training_pairs.append(pair)
            
            # Generate negative pairs (different identities)
            identity_ids = list(identity_detections.keys())
            for i in range(len(identity_ids)):
                for j in range(i + 1, len(identity_ids)):
                    id1, id2 = identity_ids[i], identity_ids[j]
                    dets1, dets2 = identity_detections[id1], identity_detections[id2]
                    
                    # Sample some negative pairs
                    for _ in range(min(20, len(dets1) * len(dets2))):
                        det1 = random.choice(dets1)
                        det2 = random.choice(dets2)
                        
                        pair = {
                            'crop1_path': det1['crop_path'],
                            'crop2_path': det2['crop_path'],
                            'label': 0,  # Different identity
                            'identity_id': -1,  # Invalid for negative pairs
                            'track1_id': det1['track_id'],
                            'track2_id': det2['track_id'],
                            'frame1_id': det1['frame_id'],
                            'frame2_id': det2['frame_id'],
                            'temporal_gap': abs(det1['frame_id'] - det2['frame_id'])
                        }
                        training_pairs.append(pair)
        
        # Shuffle pairs
        random.shuffle(training_pairs)
        
        # Save training pairs
        pairs_path = self.output_dir / "training_pairs.json"
        with open(pairs_path, 'w') as f:
            json.dump(training_pairs, f, indent=2)
        
        print(f"Generated {len(training_pairs)} training pairs")
        return training_pairs
    
    def generate_pair(self) -> Dict:
        """
        Generate a single training pair for testing purposes
        
        Returns:
            Dict containing:
                - image1: First face crop
                - image2: Second face crop  
                - temporal_gap: Time gap between frames
                - same_person: Whether faces belong to same person
        """
        # Generate two random face crops
        image1 = self._generate_random_face_crop()
        image2 = self._generate_random_face_crop()
        
        # Random temporal gap between 0.05 and 0.3 seconds
        temporal_gap = random.uniform(0.05, 0.3)
        
        # 50% chance they're the same person
        same_person = random.choice([True, False])
        
        # If same person, make the second image similar to first
        if same_person:
            # Add slight variations to simulate same person in different frames
            image2 = self._add_variations(image1)
        
        return {
            'image1': image1,
            'image2': image2,
            'temporal_gap': temporal_gap,
            'same_person': same_person
        }
    
    def _generate_random_face_crop(self) -> np.ndarray:
        """Generate a random face crop for testing"""
        # Create a random face-like image
        face = np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8)
        
        # Add some face-like features (very basic)
        # Eyes
        cv2.circle(face, (30, 35), 5, (20, 20, 20), -1)
        cv2.circle(face, (82, 35), 5, (20, 20, 20), -1)
        
        # Nose
        cv2.circle(face, (56, 55), 3, (100, 100, 100), -1)
        
        # Mouth
        cv2.ellipse(face, (56, 75), (10, 5), 0, 0, 180, (50, 50, 50), -1)
        
        return face
    
    def _add_variations(self, image: np.ndarray) -> np.ndarray:
        """Add slight variations to simulate same person in different frames"""
        varied_image = image.copy()
        
        # Add noise
        noise = np.random.normal(0, 10, image.shape)
        varied_image = np.clip(varied_image + noise, 0, 255).astype(np.uint8)
        
        # Slight brightness change
        brightness_factor = random.uniform(0.8, 1.2)
        varied_image = np.clip(varied_image * brightness_factor, 0, 255).astype(np.uint8)
        
        return varied_image

    def main():
        """Generate synthetic dataset"""
        generator = SyntheticFaceTrackGenerator(
            output_dir="synthetic_data",
            num_identities=20,
            num_videos=50,
            frames_per_video=30,
            max_faces_per_frame=3
        )
        
        # Generate dataset
        dataset_metadata = generator.generate_dataset()
        
        # Create training pairs
        training_pairs = generator.create_training_pairs(dataset_metadata)
        
        print(f"Dataset generation complete!")
        print(f"Generated {len(dataset_metadata['videos'])} videos")
        print(f"Generated {len(training_pairs)} training pairs")


    if __name__ == '__main__':
        main()
