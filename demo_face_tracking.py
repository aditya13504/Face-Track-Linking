#!/usr/bin/env python3
"""
CNN Face Track Linking Demo Script

This script demonstrates the complete CNN face track linking system using the pre-trained model.
It showcases real-time face tracking, feature extraction, and tracklet association capabilities.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import our modules
from face_track_linking_model import FaceTrackLinkingModel, TrackLinkingMetrics
from face_recognition_module import FaceRecognitionSystem
from data_loader import FaceTrackLinkingDataset

class VideoInputHandler:
    """
    Handles video input from webcam or file and extracts frames
    """
    
    def __init__(self):
        self.frames = []
        self.fps = 30
        
    def get_user_choice(self) -> str:
        """
        Get user's choice for video input method
        
        Returns:
            'webcam' or 'file'
        """
        print("\n🎥 VIDEO INPUT SELECTION")
        print("=" * 50)
        print("Choose your video input method:")
        print("1️⃣  Live webcam recording (2 minutes)")
        print("2️⃣  Use existing video file")
        print("=" * 50)
        
        while True:
            choice = input("Enter your choice (1 or 2): ").strip()
            if choice == '1':
                return 'webcam'
            elif choice == '2':
                return 'file'
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
    
    def record_from_webcam(self, duration_seconds: int = 120) -> List[np.ndarray]:
        """
        Record video from webcam for specified duration
        
        Args:
            duration_seconds: Recording duration in seconds (default: 2 minutes)
            
        Returns:
            List of frame arrays
        """
        print(f"\n📹 Starting webcam recording for {duration_seconds} seconds...")
        print("Press 'q' to stop recording early")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("❌ Could not open webcam")
        
        # Set webcam properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frames = []
        start_time = time.time()
        frame_count = 0
        
        print("🔴 Recording started... Look at the camera!")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to capture frame")
                    break
                
                # Add timestamp to frame
                elapsed_time = time.time() - start_time
                cv2.putText(frame, f"Recording: {elapsed_time:.1f}s / {duration_seconds}s", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to stop", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Display frame
                cv2.imshow('Webcam Recording', frame)
                
                # Save frame
                frames.append(frame.copy())
                frame_count += 1
                
                # Check for exit conditions
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("🛑 Recording stopped by user")
                    break
                
                if elapsed_time >= duration_seconds:
                    print(f"⏰ Recording completed ({duration_seconds} seconds)")
                    break
                    
        except KeyboardInterrupt:
            print("🛑 Recording interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        print(f"✅ Recorded {len(frames)} frames ({len(frames)/30:.1f} seconds)")
        return frames
    
    def load_from_file(self, video_path: str) -> List[np.ndarray]:
        """
        Load video from file and extract frames
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of frame arrays
        """
        print(f"\n📁 Loading video from: {video_path}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"❌ Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"❌ Could not open video file: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        
        print(f"📊 Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f} seconds")
        
        frames = []
        frame_count = 0
        
        try:
            with tqdm(total=total_frames, desc="Extracting frames") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frames.append(frame)
                    frame_count += 1
                    pbar.update(1)
                    
        finally:
            cap.release()
        
        print(f"✅ Extracted {len(frames)} frames from video")
        return frames
    
    def get_video_input(self) -> Tuple[List[np.ndarray], Dict]:
        """
        Get video input based on user choice
        
        Returns:
            Tuple of (frames_list, metadata)
        """
        choice = self.get_user_choice()
        
        if choice == 'webcam':
            frames = self.record_from_webcam()
            metadata = {
                'source': 'webcam',
                'duration': len(frames) / 30,
                'total_frames': len(frames),
                'fps': 30
            }
        else:  # file
            while True:
                video_path = input("\n📁 Enter video file path: ").strip().strip('"\'')
                try:
                    frames = self.load_from_file(video_path)
                    metadata = {
                        'source': 'file',
                        'path': video_path,
                        'total_frames': len(frames),
                        'duration': len(frames) / 30  # Assuming 30 FPS
                    }
                    break
                except (FileNotFoundError, RuntimeError) as e:
                    print(f"❌ Error: {e}")
                    retry = input("Try again? (y/n): ").strip().lower()
                    if retry != 'y':
                        raise SystemExit("Exiting...")
        
        return frames, metadata

class FaceTrackLinkingDemo:
    """
    Demonstration class for CNN Face Track Linking System
    """
    
    def __init__(self, checkpoint_path: str = "checkpoints/face_track_linking_model.pth"):
        """
        Initialize the demo system
        
        Args:
            checkpoint_path: Path to the pre-trained model checkpoint
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Initializing CNN Face Track Linking Demo on {self.device}")
        
        # Load configuration
        with open('train_config.json', 'r') as f:
            self.config = json.load(f)
        
        # Initialize model
        self.model = self._load_pretrained_model(checkpoint_path)
        self.metrics = TrackLinkingMetrics()
        
        # Initialize face recognition system
        try:
            self.face_system = FaceRecognitionSystem()
            print("✅ Face recognition system loaded successfully")
        except Exception as e:
            print(f"⚠️ Face recognition system not available: {e}")
            self.face_system = None
        
        # Initialize video input handler
        self.video_handler = VideoInputHandler()
        
        # Demo statistics
        self.demo_stats = {
            'total_tracklets_processed': 0,
            'successful_links': 0,
            'processing_times': [],
            'confidence_scores': []
        }
        
        print("🎯 CNN Face Track Linking Demo initialized successfully!")
        self._print_model_info()
    
    def _load_pretrained_model(self, checkpoint_path: str) -> FaceTrackLinkingModel:
        """Load the pre-trained model"""
        print(f"📦 Loading pre-trained model from {checkpoint_path}")
        
        # Create model with config
        model = FaceTrackLinkingModel(
            backbone=self.config['model']['backbone'],
            embedding_dim=self.config['model']['embedding_dim'],
            max_distance=self.config['model']['max_distance']
        ).to(self.device)
        
        # Load checkpoint
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Try to load state dict with strict=False to handle architecture differences
                if 'model_state_dict' in checkpoint:
                    missing_keys, unexpected_keys = model.load_state_dict(
                        checkpoint['model_state_dict'], strict=False
                    )
                    print(f"✅ Model loaded with {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys")
                    
                    # Print training metrics if available
                    if 'train_loss' in checkpoint:
                        print(f"📊 Training Loss: {checkpoint['train_loss']:.4f}")
                    if 'val_loss' in checkpoint:
                        print(f"📊 Validation Loss: {checkpoint['val_loss']:.4f}")
                    if 'epoch' in checkpoint:
                        print(f"📊 Trained for {checkpoint['epoch']} epochs")
                else:
                    # Try loading direct state dict
                    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
                    print(f"✅ Model loaded with {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys")
                
                print("⚠️  Note: Some parameters may be randomly initialized due to architecture differences")
                
            except Exception as e:
                print(f"⚠️ Error loading checkpoint: {e}")
                print("🔄 Using randomly initialized model for demonstration")
        else:
            print(f"⚠️ Checkpoint not found at {checkpoint_path}")
            print("🔄 Using randomly initialized model for demonstration")
        
        model.eval()
        return model
    
    def _print_model_info(self):
        """Print model architecture information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print("\n🏗️  MODEL ARCHITECTURE INFO")
        print("=" * 50)
        print(f"Backbone: {self.config['model']['backbone']}")
        print(f"Embedding Dimension: {self.config['model']['embedding_dim']}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Model Size: {total_params * 4 / (1024**2):.2f} MB")
        print("=" * 50)
    
    def generate_synthetic_tracklets(self, num_tracklets: int = 10) -> List[Dict]:
        """
        Generate synthetic tracklets for demonstration
        
        Args:
            num_tracklets: Number of tracklets to generate
            
        Returns:
            List of synthetic tracklet data
        """
        print(f"🎭 Generating {num_tracklets} synthetic tracklets for demo...")
        
        tracklets = []
        for i in range(num_tracklets):
            # Generate synthetic tracklet sequence
            seq_len = np.random.randint(3, 8)  # Variable sequence length
            
            tracklet = {
                'id': i,
                'sequence_length': seq_len,
                'images': torch.randn(seq_len, 3, 224, 224),  # Synthetic face crops
                'bboxes': torch.randn(seq_len, 4),  # Synthetic bounding boxes
                'timestamps': torch.arange(seq_len).float(),
                'confidence': torch.rand(seq_len),
                'identity_id': np.random.randint(0, 5) if i < 8 else np.random.randint(0, 15)  # Some overlap
            }
            tracklets.append(tracklet)
        
        print(f"✅ Generated {len(tracklets)} synthetic tracklets")
        return tracklets
    
    def process_tracklet_pair(self, tracklet1: Dict, tracklet2: Dict) -> Dict:
        """
        Process a pair of tracklets to determine if they should be linked
        
        Args:
            tracklet1: First tracklet
            tracklet2: Second tracklet
            
        Returns:
            Dictionary with linking results
        """
        start_time = time.time()
        
        with torch.no_grad():
            # Use middle frames from each tracklet for comparison
            mid_idx1 = tracklet1['sequence_length'] // 2
            mid_idx2 = tracklet2['sequence_length'] // 2
            
            crop1 = tracklet1['images'][mid_idx1:mid_idx1+1].to(self.device)
            crop2 = tracklet2['images'][mid_idx2:mid_idx2+1].to(self.device)
            
            # Forward pass through model
            outputs = self.model(crop1, crop2)
            
            # Get similarity score
            similarity = torch.sigmoid(outputs['similarity_score']).cpu().item()
            
            # Determine if tracklets should be linked
            should_link = similarity > 0.5
            
            processing_time = time.time() - start_time
            
            return {
                'tracklet1_id': tracklet1['id'],
                'tracklet2_id': tracklet2['id'],
                'similarity_score': similarity,
                'should_link': should_link,
                'processing_time': processing_time,
                'features1': outputs['features1'].cpu(),
                'features2': outputs['features2'].cpu(),
                'ground_truth': tracklet1['identity_id'] == tracklet2['identity_id']
            }
    
    def run_tracklet_association_demo(self, tracklets: List[Dict]) -> Dict:
        """
        Run complete tracklet association demonstration
        
        Args:
            tracklets: List of tracklets to process
            
        Returns:
            Results summary
        """
        print(f"\n🔗 Running Tracklet Association Demo on {len(tracklets)} tracklets")
        print("=" * 60)
        
        results = []
        correct_predictions = 0
        total_pairs = 0
        
        # Process all pairs of tracklets
        for i in tqdm(range(len(tracklets)), desc="Processing tracklets"):
            for j in range(i + 1, len(tracklets)):
                result = self.process_tracklet_pair(tracklets[i], tracklets[j])
                results.append(result)
                
                # Update statistics
                self.demo_stats['total_tracklets_processed'] += 1
                self.demo_stats['processing_times'].append(result['processing_time'])
                self.demo_stats['confidence_scores'].append(result['similarity_score'])
                
                if result['should_link']:
                    self.demo_stats['successful_links'] += 1
                
                # Check accuracy
                if result['should_link'] == result['ground_truth']:
                    correct_predictions += 1
                total_pairs += 1
        
        accuracy = correct_predictions / total_pairs if total_pairs > 0 else 0
        
        # Generate summary
        summary = {
            'total_pairs_processed': total_pairs,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'average_processing_time': np.mean(self.demo_stats['processing_times']),
            'average_confidence': np.mean(self.demo_stats['confidence_scores']),
            'successful_links': self.demo_stats['successful_links'],
            'detailed_results': results
        }
        
        self._print_results_summary(summary)
        return summary
    
    def _print_results_summary(self, summary: Dict):
        """Print formatted results summary"""
        print(f"\n📊 TRACKLET ASSOCIATION RESULTS")
        print("=" * 60)
        print(f"Total Pairs Processed: {summary['total_pairs_processed']}")
        print(f"Correct Predictions: {summary['correct_predictions']}")
        print(f"Accuracy: {summary['accuracy']:.2%}")
        print(f"Successful Links: {summary['successful_links']}")
        print(f"Average Processing Time: {summary['average_processing_time']*1000:.2f} ms")
        print(f"Average Confidence Score: {summary['average_confidence']:.3f}")
        print("=" * 60)
    
    def visualize_results(self, results: Dict):
        """
        Create visualizations of the demo results
        
        Args:
            results: Results from tracklet association demo
        """
        print("📈 Creating result visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('CNN Face Track Linking Demo Results', fontsize=16, fontweight='bold')
        
        # 1. Confidence Score Distribution
        confidences = [r['similarity_score'] for r in results['detailed_results']]
        axes[0, 0].hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(0.5, color='red', linestyle='--', label='Decision Threshold')
        axes[0, 0].set_title('Similarity Score Distribution')
        axes[0, 0].set_xlabel('Similarity Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Processing Time Analysis
        times = np.array(self.demo_stats['processing_times']) * 1000  # Convert to ms
        axes[0, 1].plot(times, marker='o', linewidth=2, markersize=4, color='green')
        axes[0, 1].set_title('Processing Time per Pair')
        axes[0, 1].set_xlabel('Pair Index')
        axes[0, 1].set_ylabel('Processing Time (ms)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Accuracy by Confidence Range
        conf_ranges = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
        range_labels = ['Low\n(0-0.3)', 'Med-Low\n(0.3-0.5)', 'Med-High\n(0.5-0.7)', 'High\n(0.7-1.0)']
        accuracies = []
        
        for low, high in conf_ranges:
            range_results = [r for r in results['detailed_results'] 
                           if low <= r['similarity_score'] < high]
            if range_results:
                correct = sum(1 for r in range_results if r['should_link'] == r['ground_truth'])
                accuracy = correct / len(range_results)
            else:
                accuracy = 0
            accuracies.append(accuracy)
        
        bars = axes[1, 0].bar(range_labels, accuracies, color=['lightcoral', 'orange', 'lightgreen', 'darkgreen'])
        axes[1, 0].set_title('Accuracy by Confidence Range')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{acc:.2%}', ha='center', va='bottom')
        
        # 4. Model Performance Summary
        axes[1, 1].axis('off')
        performance_text = f"""
        🎯 MODEL PERFORMANCE SUMMARY
        
        Total Tracklet Pairs: {results['total_pairs_processed']}
        Overall Accuracy: {results['accuracy']:.2%}
        Successful Links: {results['successful_links']}
        
        ⚡ SPEED METRICS
        Avg Processing Time: {results['average_processing_time']*1000:.2f} ms
        Throughput: {1/results['average_processing_time']:.1f} pairs/sec
        
        🧠 MODEL SPECS
        Backbone: {self.config['model']['backbone']}
        Embedding Dim: {self.config['model']['embedding_dim']}
        Parameters: {sum(p.numel() for p in self.model.parameters()):,}
        """
        axes[1, 1].text(0.1, 0.9, performance_text, transform=axes[1, 1].transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        
        # Save the plot
        output_path = "outputs/demo_results.png"
        os.makedirs("outputs", exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to {output_path}")
        plt.show()
    
    def run_complete_demo(self, use_video_input: bool = True):
        """
        Run the complete CNN face track linking demonstration
        
        Args:
            use_video_input: Whether to use video input or synthetic data
        """
        print("🚀 Starting Complete CNN Face Track Linking Demo")
        print("=" * 70)
        
        if use_video_input:
            # Get video input from user
            frames, video_metadata = self.video_handler.get_video_input()
            
            print(f"\n📊 Video Metadata:")
            for key, value in video_metadata.items():
                print(f"   {key}: {value}")
            
            # Extract tracklets from frames
            tracklets = self.extract_face_tracklets_from_frames(frames)
        else:
            # Generate synthetic data
            tracklets = self.generate_synthetic_tracklets(15)
        
        # Run tracklet association
        results = self.run_tracklet_association_demo(tracklets)
        
        # Create visualizations
        self.visualize_results(results)
        
        # Generate detailed report
        self._generate_demo_report(results, tracklets)
        
        print("\n🎉 Demo completed successfully!")
        print(f"📄 Check outputs/demo_report.json for detailed results")
        return results
    
    def extract_face_tracklets_from_frames(self, frames: List[np.ndarray]) -> List[Dict]:
        """
        Extract face tracklets from video frames
        
        Args:
            frames: List of video frames
            
        Returns:
            List of tracklet dictionaries
        """
        print(f"\n🔍 Extracting face tracklets from {len(frames)} frames...")
        
        if self.face_system is None:
            print("⚠️ Face recognition system not available, using synthetic tracklets")
            return self.generate_synthetic_tracklets()
        
        tracklets = []
        face_tracks = {}  # Track faces across frames
        next_track_id = 0
        
        # Process frames in batches for efficiency
        batch_size = 10
        
        for batch_start in tqdm(range(0, len(frames), batch_size), desc="Processing frame batches"):
            batch_end = min(batch_start + batch_size, len(frames))
            
            for frame_idx in range(batch_start, batch_end):
                frame = frames[frame_idx]
                  # Detect and recognize faces in frame
                face_results = self.face_system.detect_and_recognize(frame)
                
                if not face_results:
                    continue
                
                # Extract face crops and encodings
                for i, result in enumerate(face_results):
                    bbox = result['bbox']
                    x, y, w, h = bbox
                    
                    # Extract face crop
                    face_crop = frame[y:y+h, x:x+w]
                    
                    if face_crop.size == 0:
                        continue
                    
                    # Resize to model input size
                    face_crop_resized = cv2.resize(face_crop, (224, 224))
                    face_tensor = torch.from_numpy(face_crop_resized).permute(2, 0, 1).float() / 255.0
                    
                    # Convert bbox format for tracking
                    top, left, bottom, right = y, x, y+h, x+w
                    
                    # Try to match with existing tracks
                    matched_track = None
                    if face_tracks:
                        # Simple matching based on overlap (in real implementation, use more sophisticated tracking)
                        for track_id, track_data in face_tracks.items():
                            last_bbox = track_data['bboxes'][-1]
                            # Calculate IoU or distance
                            current_center = ((left + right) / 2, (top + bottom) / 2)
                            last_center = ((last_bbox[1] + last_bbox[3]) / 2, (last_bbox[0] + last_bbox[2]) / 2)
                            
                            distance = np.sqrt((current_center[0] - last_center[0])**2 + 
                                             (current_center[1] - last_center[1])**2)
                            
                            if distance < 100:  # Threshold for same person
                                matched_track = track_id
                                break
                    
                    if matched_track is not None:
                        # Update existing track
                        face_tracks[matched_track]['images'].append(face_tensor)
                        face_tracks[matched_track]['bboxes'].append([top, left, bottom, right])
                        face_tracks[matched_track]['timestamps'].append(frame_idx)
                    else:
                        # Create new track
                        face_tracks[next_track_id] = {
                            'images': [face_tensor],
                            'bboxes': [[top, left, bottom, right]],
                            'timestamps': [frame_idx],
                            'identity_id': next_track_id  # Simple identity assignment
                        }
                        next_track_id += 1
        
        # Convert tracks to tracklets
        for track_id, track_data in face_tracks.items():
            if len(track_data['images']) >= 3:  # Minimum tracklet length
                tracklet = {
                    'id': track_id,
                    'sequence_length': len(track_data['images']),
                    'images': torch.stack(track_data['images']),
                    'bboxes': torch.tensor(track_data['bboxes']),
                    'timestamps': torch.tensor(track_data['timestamps']).float(),
                    'confidence': torch.ones(len(track_data['images'])),
                    'identity_id': track_data['identity_id']
                }
                tracklets.append(tracklet)
        
        print(f"✅ Extracted {len(tracklets)} tracklets from video frames")
        return tracklets if tracklets else self.generate_synthetic_tracklets()
    
    def _generate_demo_report(self, results: Dict, tracklets: List[Dict]):
        """Generate a comprehensive demo report"""
        report = {
            'demo_metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'device': str(self.device),
                'model_config': self.config,
                'num_tracklets': len(tracklets)
            },
            'performance_metrics': {
                'accuracy': results['accuracy'],
                'total_pairs': results['total_pairs_processed'],
                'successful_links': results['successful_links'],
                'average_processing_time_ms': results['average_processing_time'] * 1000,
                'throughput_pairs_per_second': 1 / results['average_processing_time']
            },
            'tracklet_statistics': {
                'total_tracklets': len(tracklets),
                'average_sequence_length': np.mean([t['sequence_length'] for t in tracklets]),
                'unique_identities': len(set(t['identity_id'] for t in tracklets))
            },
            'detailed_results': results['detailed_results']
        }
        
        # Save report
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/demo_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Detailed demo report saved to outputs/demo_report.json")


def main():
    """Main demo function"""
    print("🎬 CNN Face Track Linking System Demo")
    print("=====================================")
    
    # Get user choice for demo mode
    print("\n🎯 DEMO MODE SELECTION")
    print("=" * 50)
    print("Choose demo mode:")
    print("1️⃣  Video input (webcam or file)")
    print("2️⃣  Synthetic data demo")
    print("=" * 50)
    
    while True:
        mode_choice = input("Enter your choice (1 or 2): ").strip()
        if mode_choice == '1':
            use_video_input = True
            break
        elif mode_choice == '2':
            use_video_input = False
            break
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")
    
    # Initialize demo system
    demo = FaceTrackLinkingDemo()
    
    # Run complete demonstration
    results = demo.run_complete_demo(use_video_input=use_video_input)
    
    print("\n✨ Demo completed! Key highlights:")
    print(f"   • Processed {results['total_pairs_processed']} tracklet pairs")
    print(f"   • Achieved {results['accuracy']:.1%} accuracy")
    print(f"   • Average processing: {results['average_processing_time']*1000:.1f}ms per pair")
    print(f"   • Found {results['successful_links']} successful tracklet links")


if __name__ == "__main__":
    main()