# AlphaAssistive

**AlphaAssistive** is a thesis project designed to assist individuals with disabilities or visual impairments in object retrieval tasks. It combines visual and tactile perception through a collaborative human-robot interaction system. The project utilizes a combination of **Intel RealSense**, **MediaPipe**, and **Dynamixel motors** to detect a red-colored object, track its position in 3D space, guide the user’s hand alignment with voice feedback, and detect a grasping gesture to initiate the robot's retrieval action.

---

## Features

* Voice command-based color selection (e.g., "aku ingin ambil objek merah")
* Real-time object detection and tracking using Intel RealSense and OpenCV
* Hand tracking and gesture recognition using MediaPipe
* Dynamixel motor control for camera alignment
* Voice feedback using `pyttsx3`
* Depth alignment guidance between hand and object

---

## System Requirements

* Python 3.10+
* Intel RealSense D455 camera
* Dynamixel XL430-W250-T motors
* Microphone
* Compatible USB ports (for RealSense and U2D2)
* Windows OS (tested on Windows 10)

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/AlphaAssistive.git
   cd AlphaAssistive
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Example dependencies:

   ```text
   opencv-python
   numpy
   pyrealsense2
   pyttsx3
   dynamixel-sdk
   mediapipe
   SpeechRecognition
   ```

---

## How It Works

1. **Voice Input**: User gives a voice command specifying the color of the object they want to pick.
2. **Object Detection**: The robot uses a RealSense D455 RGB-D camera and OpenCV HSV thresholding to detect the object by color.
3. **Motor Alignment**: Dynamixel motors adjust the camera angle to center the object in the frame.
4. **Depth Estimation**: Calculates the distance of the object to prepare for hand alignment.
5. **Hand Tracking**: MediaPipe detects hand landmarks and tracks hand movement in 3D.
6. **Feedback System**: Voice feedback guides the user to align their palm with the object.
7. **Grasp Detection**: Once the hand is aligned and closed, the system confirms object retrieval.

---

## Folder Structure

```
AlphaAssistive/
├── main.py             
├── LICENSE              
├── README.md            
├── requirements.txt    

```

---

## Acknowledgments

* This project was developed as a final thesis by **Alfa Noora Fithria** at Universitas Airlangga, Department of Robotics and Artificial Intelligence Engineering.
* Special thanks to thesis supervisors and peers who contributed insight and support.

---

## Contact

For questions or collaboration, please contact: \[[alfa271200@gmail.com](mailto:alfa271200@gmail.com)]

---

## Demo

> Coming soon: video demo of AlphaAssistive in action.

---
