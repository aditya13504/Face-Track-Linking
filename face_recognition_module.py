"""
Advanced Face Recognition Module for Face Track Linking

This module provides high-quality face detection and recognition capabilities
using the face_recognition library with dlib backend.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import torchvision.transforms as transforms
from torchvision.models import resnet50, mobilenet_v3_large
import math
import face_recognition
import warnings
from collections import defaultdict
import pickle
import os

# Suppress the pkg_resources deprecation warning
warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")


class AdvancedFaceDetector:
    """
    High-quality face detector using face_recognition library
    """
    
    def __init__(self, model: str = "hog"):
        """
        Initialize face detector
        
        Args:
            model: Detection model - "hog" (faster) or "cnn" (more accurate)
        """
        self.model = model
        print(f"Initialized face detector with {model} model")
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image using face_recognition library
        
        Args:
            image: Input image as numpy array (BGR format)
        
        Returns:
            List of face bounding boxes as (x, y, w, h)
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image, model=self.model)
        
        # Convert from (top, right, bottom, left) to (x, y, w, h)
        bboxes = []
        for top, right, bottom, left in face_locations:
            x, y = left, top
            w, h = right - left, bottom - top
            bboxes.append((x, y, w, h))
        
        return bboxes
    
    def extract_face_crops(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """
        Extract face crops from detected faces
        
        Args:
            image: Input image (BGR format)
            faces: List of face bounding boxes
        
        Returns:
            List of face crop images (RGB format)
        """
        face_crops = []
        
        for x, y, w, h in faces:
            # Add some padding
            padding = 0.1
            pad_w = int(w * padding)
            pad_h = int(h * padding)
            
            # Calculate padded coordinates
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(image.shape[1], x + w + pad_w)
            y2 = min(image.shape[0], y + h + pad_h)
            
            # Extract face crop
            face_crop = image[y1:y2, x1:x2]
            
            if face_crop.size > 0:
                # Convert to RGB and resize
                face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                face_crop_resized = cv2.resize(face_crop_rgb, (224, 224))
                face_crops.append(face_crop_resized)
        
        return face_crops


class FaceEncodingExtractor:
    """
    Face encoding extractor using face_recognition library
    """
    
    def __init__(self, model: str = "large"):
        """
        Initialize face encoding extractor
        
        Args:
            model: Encoding model - "small" or "large"
        """
        self.model = model
        print(f"Initialized face encoding extractor with {model} model")
    
    def extract_encoding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face encoding from a face image
        
        Args:
            face_image: Face image as numpy array (RGB format)
        
        Returns:
            Face encoding as numpy array (128-dimensional) or None if no face found
        """
        # Get face encodings
        encodings = face_recognition.face_encodings(
            face_image, 
            model=self.model,
            num_jitters=1
        )
        
        if len(encodings) > 0:
            return encodings[0]
        return None
    
    def extract_encodings_from_locations(self, 
                                       image: np.ndarray, 
                                       face_locations: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """
        Extract face encodings for known face locations
        
        Args:
            image: Input image (RGB format)
            face_locations: Face locations in (top, right, bottom, left) format
        
        Returns:
            List of face encodings
        """
        encodings = face_recognition.face_encodings(
            image, 
            known_face_locations=face_locations,
            model=self.model,
            num_jitters=1
        )
        return encodings


class FaceRecognitionDatabase:
    """
    Face recognition database for storing and matching face encodings
    """
    
    def __init__(self, similarity_threshold: float = 0.6):
        self.similarity_threshold = similarity_threshold
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_metadata = []
        self.person_counter = 0
        
        # Statistics
        self.match_history = defaultdict(list)
        self.recognition_stats = {
            'total_recognitions': 0,
            'successful_matches': 0,
            'new_faces_added': 0
        }
    
    def add_face(self, encoding: np.ndarray, person_id: Optional[int] = None, metadata: Optional[Dict] = None) -> int:
        """
        Add a new face to the database
        
        Args:
            encoding: Face encoding
            person_id: Optional person ID (auto-generated if None)
            metadata: Optional metadata for the face
        
        Returns:
            Person ID
        """
        if person_id is None:
            person_id = self.person_counter
            self.person_counter += 1
        
        self.known_face_encodings.append(encoding)
        self.known_face_names.append(f"Person_{person_id}")
        self.known_face_metadata.append(metadata or {})
        
        self.recognition_stats['new_faces_added'] += 1
        
        return person_id
    
    def recognize_face(self, encoding: np.ndarray, return_confidence: bool = True) -> Tuple[Optional[int], float]:
        """
        Recognize a face from its encoding
        
        Args:
            encoding: Face encoding to recognize
            return_confidence: Whether to return confidence score
        
        Returns:
            Tuple of (person_id, confidence) or (None, 0.0) if no match
        """
        self.recognition_stats['total_recognitions'] += 1
        
        if len(self.known_face_encodings) == 0:
            return None, 0.0
        
        # Compare with known faces
        distances = face_recognition.face_distance(self.known_face_encodings, encoding)
        best_match_index = np.argmin(distances)
        best_distance = distances[best_match_index]
        
        # Convert distance to similarity (0 = identical, 1 = completely different)
        confidence = 1.0 - best_distance
        
        if confidence >= self.similarity_threshold:
            # Extract person ID from name
            person_name = self.known_face_names[best_match_index]
            person_id = int(person_name.split('_')[1])
            
            self.recognition_stats['successful_matches'] += 1
            self.match_history[person_id].append(confidence)
            
            return person_id, confidence
        
        return None, confidence
    
    def get_person_stats(self, person_id: int) -> Dict:
        """Get statistics for a specific person"""
        if person_id not in self.match_history:
            return {}
        
        matches = self.match_history[person_id]
        return {
            'total_matches': len(matches),
            'avg_confidence': np.mean(matches),
            'max_confidence': np.max(matches),
            'min_confidence': np.min(matches)
        }
    
    def save_database(self, filepath: str):
        """Save face database to file"""
        data = {
            'known_face_encodings': self.known_face_encodings,
            'known_face_names': self.known_face_names,
            'known_face_metadata': self.known_face_metadata,
            'person_counter': self.person_counter,
            'similarity_threshold': self.similarity_threshold,
            'recognition_stats': self.recognition_stats,
            'match_history': dict(self.match_history)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Face database saved to {filepath}")
    
    def load_database(self, filepath: str):
        """Load face database from file"""
        if not os.path.exists(filepath):
            print(f"Database file {filepath} not found")
            return
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.known_face_encodings = data['known_face_encodings']
        self.known_face_names = data['known_face_names']
        self.known_face_metadata = data['known_face_metadata']
        self.person_counter = data['person_counter']
        self.similarity_threshold = data.get('similarity_threshold', self.similarity_threshold)
        self.recognition_stats = data.get('recognition_stats', self.recognition_stats)
        self.match_history = defaultdict(list, data.get('match_history', {}))
        
        print(f"Face database loaded from {filepath}")
        print(f"Loaded {len(self.known_face_encodings)} known faces")


class FaceRecognitionSystem:
    """
    Complete face recognition system combining detection and recognition
    """
    
    def __init__(self, 
                 detection_model: str = "hog",
                 encoding_model: str = "large",
                 similarity_threshold: float = 0.6):
        
        self.similarity_threshold = similarity_threshold
        
        # Initialize components
        self.detector = AdvancedFaceDetector(model=detection_model)
        self.encoder = FaceEncodingExtractor(model=encoding_model)
        self.database = FaceRecognitionDatabase(similarity_threshold=similarity_threshold)
        
        print(f"Face Recognition System initialized:")
        print(f"  Detection model: {detection_model}")
        print(f"  Encoding model: {encoding_model}")
        print(f"  Similarity threshold: {similarity_threshold}")
    
    def detect_and_recognize(self, image: np.ndarray) -> List[Dict]:
        """
        Detect and recognize faces in an image
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            List of detection results with format:
            [{'bbox': (x, y, w, h), 'person_id': int, 'confidence': float, 'encoding': np.ndarray}]
        """
        # Convert to RGB for face_recognition library
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_bboxes = self.detector.detect_faces(image)
        
        # Convert bboxes to face_recognition format (top, right, bottom, left)
        face_locations = []
        for x, y, w, h in face_bboxes:
            top, right, bottom, left = y, x + w, y + h, x
            face_locations.append((top, right, bottom, left))
        
        # Extract encodings
        encodings = self.encoder.extract_encodings_from_locations(rgb_image, face_locations)
        
        results = []
        
        for bbox, encoding in zip(face_bboxes, encodings):
            # Recognize face
            person_id, confidence = self.database.recognize_face(encoding)
            
            # If no match found, add as new person
            if person_id is None:
                person_id = self.database.add_face(encoding)
                confidence = 1.0  # Perfect confidence for new face
            
            results.append({
                'bbox': bbox,
                'person_id': person_id,
                'confidence': confidence,
                'encoding': encoding
            })
        
        return results
    
    def add_person_from_images(self, images: List[np.ndarray], person_id: Optional[int] = None) -> int:
        """
        Add a person to the database using multiple images
        
        Args:
            images: List of images containing the person's face
            person_id: Optional person ID
        
        Returns:
            Person ID
        """
        encodings = []
        
        for image in images:
            # Convert to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract encoding
            encoding = self.encoder.extract_encoding(rgb_image)
            if encoding is not None:
                encodings.append(encoding)
        
        if not encodings:
            raise ValueError("No faces found in provided images")
        
        # Add first encoding to database
        final_person_id = self.database.add_face(encodings[0], person_id)
        
        # Add additional encodings for the same person
        for encoding in encodings[1:]:
            self.database.add_face(encoding, final_person_id)
        
        print(f"Added person {final_person_id} with {len(encodings)} face encodings")
        return final_person_id
    
    def get_recognition_stats(self) -> Dict:
        """Get recognition statistics"""
        return {
            'database_stats': {
                'total_known_faces': len(self.database.known_face_encodings),
                'unique_persons': self.database.person_counter
            },
            'recognition_stats': self.database.recognition_stats.copy()
        }
    
    def save_system(self, filepath: str):
        """Save the entire face recognition system"""
        self.database.save_database(filepath)
    
    def load_system(self, filepath: str):
        """Load a saved face recognition system"""
        self.database.load_database(filepath)


def demo_face_recognition_advanced():
    """
    Advanced demo of the face recognition system
    """
    print("🎯 Advanced Face Recognition System Demo")
    print("=" * 50)
    
    # Initialize face recognition system
    face_system = FaceRecognitionSystem(
        detection_model="hog",  # Use "cnn" for higher accuracy but slower speed
        encoding_model="large",
        similarity_threshold=0.6
    )
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("\nAdvanced Face Recognition Demo Controls:")
    print("  'q' - Quit")
    print("  's' - Save face database")
    print("  'l' - Load face database")
    print("  'r' - Show recognition statistics")
    print("  'c' - Clear database")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 5th frame for better performance
            if frame_count % 5 == 0:
                # Detect and recognize faces
                results = face_system.detect_and_recognize(frame)
                
                # Draw results
                for result in results:
                    x, y, w, h = result['bbox']
                    person_id = result['person_id']
                    confidence = result['confidence']
                    
                    # Choose color based on person_id
                    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                             (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)]
                    color = colors[person_id % len(colors)]
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # Draw label
                    label = f"Person {person_id} ({confidence:.2f})"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                                 (x + label_size[0], y), color, -1)
                    cv2.putText(frame, label, (x, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show statistics
                stats = face_system.get_recognition_stats()
                stats_text = [
                    f"Frame: {frame_count}",
                    f"Known Faces: {stats['database_stats']['total_known_faces']}",
                    f"Unique Persons: {stats['database_stats']['unique_persons']}",
                    f"Total Recognitions: {stats['recognition_stats']['total_recognitions']}"
                ]
                
                y_offset = 30
                for text in stats_text:
                    cv2.putText(frame, text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_offset += 25
            
            # Display frame
            cv2.imshow('Advanced Face Recognition Demo', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                face_system.save_system('face_recognition_database.pkl')
                print("Face database saved!")
            elif key == ord('l'):
                face_system.load_system('face_recognition_database.pkl')
                print("Face database loaded!")
            elif key == ord('r'):
                stats = face_system.get_recognition_stats()
                print("\n=== Recognition Statistics ===")
                print(f"Known faces: {stats['database_stats']['total_known_faces']}")
                print(f"Unique persons: {stats['database_stats']['unique_persons']}")
                print(f"Total recognitions: {stats['recognition_stats']['total_recognitions']}")
                print(f"Successful matches: {stats['recognition_stats']['successful_matches']}")
                print(f"New faces added: {stats['recognition_stats']['new_faces_added']}")
            elif key == ord('c'):
                face_system.database = FaceRecognitionDatabase(face_system.similarity_threshold)
                print("Face database cleared!")
    
    except KeyboardInterrupt:
        print("Demo interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Show final statistics
        stats = face_system.get_recognition_stats()
        print("\n=== Final Statistics ===")
        print(f"Total frames processed: {frame_count}")
        print(f"Known faces in database: {stats['database_stats']['total_known_faces']}")
        print(f"Unique persons identified: {stats['database_stats']['unique_persons']}")


if __name__ == "__main__":
    demo_face_recognition_advanced()
