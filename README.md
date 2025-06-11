# CNN Face Track Linking System

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)

A robust face tracking system based on CNN technology for tracking and re-identifying faces across video frames. This system combines state-of-the-art face detection, deep feature extraction, and tracklet association techniques based on research insights from multiple academic papers.

## 📋 Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Workflow](#-workflow)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Demo](#running-the-demo)
  - [Training Your Own Model](#training-your-own-model)
  - [Generating Synthetic Data](#generating-synthetic-data)
- [Components](#components)
- [Research Background](#research-background)
- [Configuration](#configuration)
- [Results](#results)
- [Contributing](#contributing)

## ✨ Features

- **Advanced Face Detection**: Robust face detection using either HOG-based or CNN-based detector from the face_recognition library
- **Deep Feature Extraction**: Extracts discriminative features from faces using ResNet-50 or VGG-16 backbone
- **Face Recognition**: Support for face recognition to identify known individuals
- **Temporal Association**: Links face tracklets across frames using a combination of appearance and motion cues
- **Online & Semi-Online Modes**: Supports both real-time tracking and batch processing
- **Visualization Tools**: Includes visualization tools for face tracks and association results
- **Synthetic Data Generation**: Can generate synthetic training data when real annotated data is not available
- **Extensible Architecture**: Modular design allows for easy extension and experimentation

## 🏗️ System Architecture

The system consists of the following main components:

1. **Video Input Handler**: Processes video from webcam or file sources
2. **Face Detection & Recognition**: Detects faces in frames and extracts identity features
3. **Deep Feature Extraction**: Extracts discriminative appearance features
4. **Track Linking Model**: Associates face tracklets across frames using deep learning
5. **Visualization Module**: Renders tracking results with bounding boxes and track IDs

**Processing Pipeline**:

```
Video Input → Face Detection → Feature Extraction → Track Association → Visualization
```

## 🔄 Workflow

### 1. Input Processing and Face Detection
- **Video Ingestion**: The system accepts input from either a webcam feed or a pre-recorded video file
- **Frame Extraction**: Individual frames are extracted from the video stream at the configured frame rate
- **Face Detection**: Each frame is processed using the `AdvancedFaceDetector` to detect and localize faces:
  - HOG-based detection (faster) or CNN-based detection (more accurate) is applied
  - Each detected face results in a bounding box (x, y, width, height)
  - Additional face landmarks can be extracted for alignment and normalization

### 2. Feature Extraction and Recognition
- **Face Alignment**: Detected faces are aligned using facial landmarks to normalize pose
- **Deep Feature Extraction**: The `DeepFeatureExtractor` extracts embeddings from each face:
  - ResNet-50/VGG-16 backbone networks convert face crops to 512-dimensional embeddings
  - These embeddings represent the distinctive appearance of each face
- **Identity Recognition**: For known faces, the `FaceRecognitionSystem` attempts to match against the database:
  - Face embeddings are compared against known identities
  - If similarity exceeds the configured threshold, an identity is assigned

### 3. Short-term Tracking
- **Motion-based Tracking**: Frame-to-frame tracking is performed using motion cues:
  - Kalman filtering predicts the next position of each tracked face
  - IoU (Intersection over Union) matching associates detections with existing tracks
  - New tracks are initialized for unmatched detections
  - Tracks without matches for several frames are marked as inactive

### 4. Tracklet Association
- **Tracklet Formation**: Continuous frame-to-frame tracks form tracklets
- **Feature Aggregation**: For each tracklet, face features are aggregated over time:
  - Temporal average embedding is computed for stable representation
  - Quality assessment determines reliable feature frames
- **Track Linking**: The `FaceTrackLinkingModel` links separated tracklets:
  - Deep appearance similarity between tracklet pairs is computed
  - Motion consistency between tracklet endpoints is evaluated
  - Combined affinity scores determine linking decisions
  - Hungarian algorithm optimally solves the tracklet association problem

### 5. Track Management
- **ID Assignment**: Each track is assigned a unique ID
- **Long-term Re-identification**:
  - When faces disappear and reappear, the system attempts to link to previous tracks
  - Historical embeddings are used for matching across larger temporal gaps
- **Occlusion Handling**: The system bridges gaps when faces are temporarily occluded:
  - Position is interpolated during short occlusions
  - Appearance similarity resolves identity after longer occlusions

### 6. Visualization and Output
- **Bounding Box Rendering**: Each tracked face is displayed with its bounding box
- **Track ID Visualization**: Consistent IDs are shown for each tracked face
- **Trajectory Visualization**: Optional visualization of movement patterns
- **Results Export**: Tracking results can be exported to JSON for further analysis:
  - Face locations over time
  - Track IDs and associations
  - Recognition results when available

### 7. Training Workflow (Offline)
- **Data Preparation**: Training data is prepared using:
  - Real annotated video sequences, or
  - Synthetic data generation for specific scenarios
- **Pair Sampling**: Positive and negative tracklet pairs are sampled:
  - Same-identity tracklets with temporal gaps (positives)
  - Different-identity tracklets (negatives)
- **Model Training**: The CNN track linking model is trained using:
  - Triplet loss to separate same/different identity embeddings
  - Contrastive loss to enforce consistent temporal links
  - Motion consistency constraints as a regularizer
- **Evaluation**: Model performance is evaluated on validation data:
  - Association accuracy metrics
  - Identity preservation metrics
  - Confusion matrix analysis

## 🚀 Installation

1. Clone this repository:

```bash
git clone https://github.com/aditya13504/Face-Track-Linking.git
cd Face-Track-Linking
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download pre-trained models (automatically done on first run) or place your own models in the `checkpoints/` directory.

**Note**: The system requires the following dependencies:
- PyTorch (2.0+)
- OpenCV (4.8+)
- face_recognition (1.3+)
- dlib (19.24+)
- NumPy, Matplotlib, and other common data science libraries

## 🎮 Usage

### Running the Demo

To run the face tracking demo on your webcam or a video file:

```bash
python demo_face_tracking.py
```

The demo will prompt you to choose between:
1. Live webcam input
2. Video file input

Follow the on-screen instructions to select your input source and configure tracking parameters.

### Training Your Own Model

To train the face track linking model on your own data:

1. Organize your training data (or use the synthetic data generator)
2. Configure training parameters in `train_config.json`
3. Run the training script:

```bash
python train_face_linking.py --config train_config.json
```

The training process will:
- Load training data and perform data augmentation
- Train the model using triplet and contrastive losses
- Evaluate the model periodically
- Save checkpoints to the `checkpoints/` directory
- Log metrics to the `logs/` directory (viewable with TensorBoard)

### Generating Synthetic Data

If you don't have annotated face tracking data, you can generate synthetic data:

```bash
python synthetic_data_generator.py --output_dir synthetic_data --num_identities 10 --num_videos 100
```

This will create synthetic face tracks with known associations that can be used for training.

## 🧩 Components

### 1. Face Recognition Module (`face_recognition_module.py`)

Provides face detection and recognition capabilities using the face_recognition library with dlib backend. Includes:
- `AdvancedFaceDetector`: Detects faces using either HOG or CNN methods
- `FaceFeatureExtractor`: Extracts appearance features from detected faces
- `FaceRecognitionSystem`: Combines detection and feature extraction

### 2. Face Track Linking Model (`face_track_linking_model.py`)

Implements the CNN-based track linking model based on research papers. Includes:
- `DeepFeatureExtractor`: CNN for extracting discriminative features
- `MotionEncoder`: Encodes relative motion between tracklets
- `FaceTrackLinkingModel`: Main model for track association
- `TrackLinkingMetrics`: Evaluation metrics for model performance

### 3. Data Loader (`data_loader.py`)

Handles loading and preprocessing of training data. Includes:
- `FaceTrackLinkingDataset`: Dataset class for track linking pairs
- Data augmentation and normalization functions
- Batch sampling strategies

### 4. Training Script (`train_face_linking.py`)

Implements the training loop and evaluation functions. Includes:
- `FaceTrackLinkingTrainer`: Manages the training process
- Loss functions (triplet loss, contrastive loss)
- Metrics tracking and visualization

### 5. Synthetic Data Generator (`synthetic_data_generator.py`)

Generates synthetic face tracking data for training. Includes:
- `SyntheticFaceTrackGenerator`: Creates synthetic face tracks with known associations
- Appearance and motion simulation

### 6. Demo Script (`demo_face_tracking.py`)

Demonstrates the system on real video input. Includes:
- `VideoInputHandler`: Processes video from webcam or files
- `FaceTrackingDemo`: Runs the complete tracking pipeline
- `ResultVisualizer`: Visualizes tracking results

## 📚 Research Background

This implementation is based on insights from several research papers in the `dataset/` directory, including:
- "A Novel Method for Face Track Linking in Videos"
- "The New High-Performance Face Tracking System based on Detection-Tracking and Tracklet-Tracklet Association in Semi-Online Mode"
- "FaceSORT: A Real-Time Face Tracking Method"

Key techniques implemented:
- Bipartite matching with Hungarian algorithm
- Tracklet-to-tracklet association using deep features
- Combined appearance and motion affinity scores
- Semi-online processing for improved accuracy

## ⚙️ Configuration

The system can be configured through the `train_config.json` file:

```json
{
  "model": {
    "backbone": "resnet50",
    "embedding_dim": 512,
    "max_distance": 0.7
  },
  "training": {
    "batch_size": 16,
    "learning_rate": 0.001,
    "num_epochs": 50,
    "triplet_margin": 1.0,
    "contrastive_margin": 2.0,
    "loss_weights": {
      "triplet": 0.4,
      "contrastive": 0.4,
      "motion": 0.2
    }
  }
}
```

## 📊 Results

The face tracking system achieves:
- High accuracy in face detection and recognition
- Robust tracking through occlusions and challenging conditions
- Fast processing suitable for real-time applications
- Reliable re-identification of faces after they reappear

Performance metrics are logged during training and can be visualized using TensorBoard:

```bash
tensorboard --logdir=logs
```

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- The face_recognition library by Adam Geitgey
- PyTorch and torchvision teams
- Authors of the research papers in the `dataset/` directory
