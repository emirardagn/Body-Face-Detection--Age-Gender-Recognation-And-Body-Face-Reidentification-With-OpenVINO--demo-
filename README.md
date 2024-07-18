## Real-Time Face and Person Reidentification
This project is designed to perform real-time face and person reidentification using OpenVINO models. The application captures video from a webcam or a video file, detects faces and bodies, estimates age and gender, and tracks the identified individuals across frames.

##Features
- Real-time video capture: Supports video input from a webcam or a pre-recorded video file.
- Face detection: Detects faces in each frame and assigns unique IDs.
- Person reidentification: Identifies and tracks persons based on body features.
- Age and gender estimation: Estimates the age and gender of detected faces.
- Pose estimation: Determines if a person is looking at the camera or not.
- Intersection over Union (IoU) calculation: Matches faces to bodies based on overlapping bounding boxes.
#Requirements
- Python 3.7+
- OpenCV
- NumPy
- SciPy
- OpenVINO Toolkit
