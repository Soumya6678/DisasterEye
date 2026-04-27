DisasterEye: Survivor Detection System 
1. Project Abstract
DisasterEye is an edge-AI based survivor detection system designed for post-disaster search and rescue scenarios. The system uses computer vision and deep learning to identify exposed human body parts (hand, head, torso, leg, foot) from rubble scenes and infer survivor presence using body-part clustering logic. The solution is designed for deployment on NVIDIA Jetson Nano for low-latency edge inference in environments where cloud connectivity may be unavailable.
________________________________________
2. Problem Statement
After earthquakes, landslides, and structural collapse, survivors may remain trapped under debris with only partial limbs visible. Traditional person detectors often fail under severe occlusion.
Problem:
Detect potential survivors from partial body visibility in rubble.
Objective:
Use edge AI to:
•	Detect exposed body parts
•	Infer survivor presence from grouped detections
•	Trigger rescue alerts in real time
________________________________________
3. Role of Edge Computing
Components executed on Jetson Nano:
•	Image capture via CSI camera
•	Preprocessing pipeline
•	YOLO inference
•	Survivor clustering logic
•	Real-time alerts
Why edge computing?
•	Low latency response
•	Works offline
•	Reduced dependency on cloud
•	Suitable for field deployment
•	Energy-efficient embedded inference
________________________________________
4. Methodology / Pipeline
System Pipeline
Input → Preprocessing → YOLO Detection → Survivor Clustering → Alert Output
Stage 1: Input
•	CSI Camera / Webcam / Video Feed
•	Rubble simulation or disaster images
Stage 2: Preprocessing
•	Contrast enhancement
•	Noise reduction
•	Frame normalization
Stage 3: Model Inference
YOLO-based detector predicts:
Classes: 1. body1 (upper torso) 2. body2 (lower torso) 3. foot 4. hand 5. head 6. leg
Total Classes = 6
Stage 4: Survivor Inference Logic
If multiple body parts occur within spatial proximity:
Examples:
•	hand + head → survivor
•	torso + leg → survivor
•	2+ unique parts clustered → survivor candidate
Stage 5: Output
•	Bounding boxes
•	Survivor count
•	Alert trigger
•	FPS display
________________________________________
5. Architectural Diagram
System Architecture
CSI Camera / Video Input
        |
        v
 Preprocessing Module
        |
        v
 YOLOv8 Body Part Detector
        |
        v
 Survivor Clustering Logic
        |
        +------> Alert System
        |
        v
 Visualization + FPS Output
Deployment Architecture
Jetson Nano
├── CSI Camera
├── Python/OpenCV Pipeline
├── YOLO Model (best.pt)
├── Survivor Logic
└── Display / Alert Output
(Figure 1: System Architecture) (Figure 2: Jetson Deployment Architecture)
________________________________________
6. Model Details
Model Used
YOLOv8 object detector
Base weights:
•	YOLOv8m (training)
•	Optimized demo inference
Framework:
•	PyTorch
•	Ultralytics
•	OpenCV
Input size:
•	Training: 960
•	Optimized inference: 320
Optimization:
•	Frame skipping
•	Reduced resolution
•	Feature disabling for FPS gain
________________________________________
7. Training Details
Dataset Used
Custom Roboflow dataset (VictimDet)
Dataset contents:
•	3794 annotated images
•	Rubble scenes
•	Human body-part annotations
Split:
•	Train: 2669
•	Valid: 752
•	Test: 373
________________________________________
8. Training Results
•	 
 ## Training Graphs

![Training Graphs](assets/training_graphs.png)

## Confusion Matrix

![Confusion Matrix](assets/confusion_matrix.png)

## Validation Predictions

![Validation Predictions](assets/val_predictions.jpg)
 
Performance
Validation showed accurate body-part localization.
Classes detected:
•	body1
•	body2
•	foot
•	hand
•	head
•	leg
________________________________________
9. Results / Output
Local System Performance
•	Optimized inference: ~16 FPS
•	Multi-survivor detection demonstrated
•	Survivor clustering working
Jetson Nano (Target)
Designed for edge deployment.
Sample Outputs
Insert:
•	Single survivor detection result
•	Two survivor demo frame
•	No-survivor negative test
 ________________________________________
10. Performance Comparison
Platform	FPS   Inference Time
Initial CPU prototype	1.3          770 ms
Optimized Local	16            143.6 ms
Jetson Nano - PyTorch
Jetson Nano-TensorRT FP16	6               165 ms
12	80 ms
________________________________________
11. Setup Instructions
Install
pip install -r requirements.txt
Run webcam
python main.py
Run demo video
python main.py --source vid_1.mp4
Test image/video
python test.py --source rubble1.jpg
________________________________________
12. Repository Structure
DisasterEye/
├── main.py
├── preprocessing.py
├── training.py
├── inference.py
├── utils.py
├── config.py
├── logger.py
├── requirements.txt
├── models/best.pt
├── assets/
└── data/disaster.yaml
________________________________________
________________________________________
13. References
1.	Ultralytics YOLOv8 Documentation.
2.	Roboflow Dataset Management Platform.
3.	NVIDIA Jetson Nano Developer Guide.
4.	Redmon, J. et al. YOLO Object Detection Papers.
5.	PyTorch Documentation.
6.	OpenCV Documentation.
(Format references per your department style if needed.)
________________________________________
14. Future Improvements
•	Thermal fusion for victim detection
•	Audio-based survivor cues
•	TensorRT optimization on Jetson
•	Segmentation-based trapped human localization

