# H2V Algorithm (Horizontal to Vertical Video Converter)

H2V is an intelligent video processing tool that automatically converts horizontal (landscape) videos to vertical (portrait) format while preserving the main subject of interest in each scene. It uses computer vision techniques to track important elements and create professional-quality vertical videos optimized for mobile platforms like Instagram Stories, TikTok, and YouTube Shorts.

## Features

- **Intelligent Subject Detection**: Automatically identifies and tracks the most important elements in each frame
- **Smart Scene Detection**: Splits videos into scenes for optimized processing of different content types
- **Speaker Focus**: Prioritizes active speakers when converting interview or dialogue content
- **Motion Tracking**: Follows moving subjects when no speaker is detected
- **Audio Preservation**: Maintains the original audio quality in the converted video
- **Smooth Transitions**: Implements intelligent focus point smoothing to prevent jarring camera movements
- **Manual Refinement Support**: Exports tracking data that can be manually adjusted if needed
- **9:16 Aspect Ratio**: Creates perfect vertical videos ready for mobile platforms

## Table of Contents

- [H2V Algorithm (Horizontal to Vertical Video Converter)](#h2v-algorithm-horizontal-to-vertical-video-converter)
  - [Features](#features)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [Usage](#usage)
    - [Basic Usage](#basic-usage)
  - [How It Works](#how-it-works)
  - [Requirements](#requirements)
  - [Configuration](#configuration)
  - [Output](#output)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)

## Installation

### Prerequisites

- Python 3.8+
- FFmpeg (required for audio processing)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/sleekmike/h2v-pipeline.git
   cd h2v-pipeline
   ```

2. Install required Python packages:
   ```bash
   
   pip install --upgrade "scenedetect[opencv]"
   pip install mediapipe==0.10.14 librosa
   OR 
   pip install -r requirements.txt
   ```

3. Install FFmpeg:
   - **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
   - **CentOS/RHEL**: `sudo yum install ffmpeg`
   - **macOS**: `brew install ffmpeg`
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

## Usage

### Basic Usage

```python
from pipeline import H2VProcessor

# Initialize the processor
processor = H2VProcessor()

# Process a video
input_path = "path/to/horizontal_video.mp4"
output_path = "path/to/vertical_video.mp4"

# Convert the video
video_path, tracking_data = processor.process_video(input_path, output_path)

print(f"Processing complete. Vertical video saved to: {video_path}")
print(f"Tracking data saved to: {tracking_data}")
```
 
## How It Works

The H2V algorithm works through a multi-stage pipeline:

1. **Scene Detection**:
   - The video is split into distinct scenes using PySceneDetect
   - Each scene is processed independently to optimize focus points

2. **Subject Identification**:
   - In each frame, the algorithm prioritizes:
     1. Speaking persons (faces + audio analysis)
     2. People in the frame (using pose estimation)
     3. Moving objects (using optical flow)

3. **Focus Point Tracking**:
   - The main subject is tracked across frames
   - Exponential smoothing is applied to prevent jerky movements

4. **Vertical Cropping**:
   - Each frame is cropped to 9:16 aspect ratio centered on the focus point
   - The algorithm ensures the most important content remains visible

5. **Audio Processing**:
   - The original audio is extracted and merged with the processed video
   - Audio synchronization is maintained throughout

6. **Tracking Data Export**:
   - Focus point coordinates are exported as JSON for potential manual refinement

## Requirements

The full list of Python dependencies:

```
opencv-python>=4.5.0
numpy>=1.19.0
scenedetect>=0.6.1
mediapipe>=0.8.9
tqdm>=4.62.0
ffmpeg-python>=0.2.0
```

A detailed `requirements.txt` file is included in the repository.

## Configuration

The H2V Processor can be configured with several parameters:

```python
processor = H2VProcessor(
    smoothing_factor=0.3,        # Controls camera movement smoothness (0.0-1.0)
    scene_threshold=30,          # Scene detection sensitivity
    face_confidence=0.5,         # Face detection confidence threshold
    pose_confidence=0.5,         # Pose detection confidence threshold
    motion_threshold=0.5,        # Motion detection sensitivity
    focus_weights={              # Relative importance of different detection methods
        'face': 1.0,
        'pose': 0.8,
        'motion': 0.6
    }
)
```

## Output

The processor generates two main outputs:

1. **Vertical Video**: A 9:16 aspect ratio video file ready for social media platforms
2. **Tracking Data**: A JSON file containing focus points with the following structure:

```json
{
  "focus_points": [
    {
      "time": 0.0,
      "original_frame": {"width": 1920, "height": 1080},
      "focus_point": {"x": 960, "y": 540},
      "crop": {"left": 720, "right": 1200, "top": 0, "bottom": 1080}
    },
    ...
  ]
}
```

This tracking data can be manually edited and used to refine the video conversion if needed.
 

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) for scene detection
- [MediaPipe](https://github.com/google/mediapipe) for face and pose detection
- [OpenCV](https://opencv.org/) for computer vision processing
- [FFmpeg](https://ffmpeg.org/) for video and audio handling

---