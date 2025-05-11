# Face Recognition with Names - Documentation

## Overview

This package provides a comprehensive face recognition system that can detect faces and identify people by name. The system includes both command-line and graphical user interface versions, allowing for flexibility in different usage scenarios.

## Features

- **Real-time Face Detection**: Detects faces in camera feed using OpenCV
- **Face Recognition with Names**: Identifies known individuals and displays their names
- **Training Interface**: Easy-to-use interface for adding new people to the recognition database
- **Persistent Database**: Stores face data and names for future recognition sessions
- **Confidence Scoring**: Shows confidence level of each recognition
- **User Management**: Add, view, and remove people from the database
- **Adjustable Settings**: Configure recognition threshold, FPS display, and more

## Requirements

- Python 3.6 or higher
- OpenCV (opencv-contrib-python)
- NumPy
- For GUI version: Tkinter and PIL (Python Imaging Library)

## Installation

1. Ensure you have Python installed on your system.
2. Install the required dependencies:

```bash
# For both versions
pip install opencv-contrib-python numpy

# For GUI version (additional dependencies)
pip install pillow
```

Note: Tkinter usually comes pre-installed with Python. If it's missing, install it using your system's package manager:

- On Ubuntu/Debian: `sudo apt-get install python3-tk`
- On Windows: Tkinter is included with standard Python installations
- On macOS: Tkinter is included with Python from python.org or can be installed via Homebrew

## Usage

### Command-Line Interface Version

Run the CLI version with:

```bash
python face_recognition_with_names.py
```

#### Controls:

- Press 'q' to quit
- Press 't' to start training mode (will prompt for name)
- Press 'c' to cancel training mode
- Press 's' to save the current recognizer
- Press 'f' to toggle FPS display
- Press 'l' to list all people in the database

### Graphical User Interface Version

Run the GUI version with:

```bash
python face_recognition_gui_with_names.py
```

The GUI provides intuitive controls for:
- Starting/stopping the camera
- Switching between recognition and training modes
- Adding new people to the database
- Viewing all people in the database
- Removing people from the database
- Adjusting recognition confidence threshold
- Toggling FPS display

## How It Works

### Face Detection

The system uses OpenCV's Haar Cascade classifier to detect faces in each frame from the camera. This is a machine learning-based approach that uses a cascade function trained with positive and negative images.

### Face Recognition

For face recognition, the system uses the Local Binary Patterns Histograms (LBPH) face recognizer from OpenCV. This algorithm:

1. Analyzes the local binary patterns in different regions of a face
2. Creates a histogram of these patterns
3. Compares new faces against known histograms to find matches

### Database System

The system maintains a persistent database of known faces:
- Each person is assigned a unique ID
- Multiple face images can be stored for each person
- Metadata like name, when added, and last seen are tracked
- Face images are stored on disk for future training sessions

### Training Process

When adding a new person:
1. The system captures multiple face images (default: 30)
2. These images are processed and added to the database
3. The face recognizer is trained with all faces in the database
4. The updated model is saved for future recognition sessions

## Customization

### Adjusting Recognition Confidence

The recognition confidence threshold determines how strict the system is when identifying faces:
- Higher threshold (e.g., 90%): Fewer false positives but might not recognize people in challenging conditions
- Lower threshold (e.g., 60%): More likely to recognize people but might have more false positives

In the CLI version, you can modify the `self.detection_confidence` value in the code.
In the GUI version, use the confidence threshold slider.

### Training More Faces

For better recognition accuracy, you can:
1. Train multiple times with the same person in different lighting conditions
2. Train with different facial expressions and angles
3. Increase the number of training images by changing `self.max_training_faces`

## Troubleshooting

### Camera Issues

If the application cannot access your camera:
1. Ensure your camera is properly connected
2. Check if another application is using the camera
3. Verify camera permissions (especially on macOS and some Linux distributions)
4. Try restarting your computer if the issue persists

### Recognition Issues

If the system fails to recognize known faces:
1. Try retraining with more images in different conditions
2. Adjust the confidence threshold
3. Ensure good lighting when using the system
4. Make sure faces are clearly visible and not partially obscured

## Advanced Usage

### Integrating with Other Systems

The face recognition components can be integrated with other systems:
- Security systems for access control
- Attendance tracking systems
- Smart home applications for personalization

### Extending Functionality

Possible extensions include:
- Emotion recognition
- Age and gender estimation
- Multiple camera support
- Cloud-based database for recognition across devices

## Files in this Package

- `face_recognition_with_names.py`: Command-line interface version
- `face_recognition_gui_with_names.py`: Graphical user interface version
- `face_database.py`: Database management system for storing face data
- `face_recognizer.yml`: Trained face recognition model (created after training)
- `face_names.pkl`: Mapping between face IDs and names (created after training)
- `known_faces/`: Directory containing stored face images (created after training)

## Credits

This application uses:
- OpenCV for computer vision capabilities
- Tkinter for the graphical user interface (GUI version)
- PIL for image processing (GUI version)
