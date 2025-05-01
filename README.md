# 🧠 Scene Understanding Pipeline using YOLOv5 and BLIP

This project implements a simple scene understanding pipeline that combines **object detection**, **image captioning**, and a basic **scene graph generation** module. It uses the following:

- YOLOv5 (Ultralytics) for object detection
- BLIP (Salesforce) for image captioning
- Mock scene graph based on spatial relationships

## 📸 Features

- Detects objects in images using a pretrained YOLOv5 model.
- Generates descriptive captions with the BLIP transformer.
- Creates basic scene graphs describing object relationships (e.g., *"dog is near person"*).
- Annotates images with bounding boxes and labels.

## 🛠 Requirements

Install the required Python packages using pip:

```bash
pip install torch torchvision torchaudio
pip install opencv-python
pip install transformers
pip install pillow
pip install git+https://github.com/ultralytics/yolov5
Project Structure
.
├── scene_understanding.py   # Main Python script
├── README.md                # This file
└── OIP.jpg                  # Sample image (or provide your own)

run the script by-
python scene_understanding.py


sample image
![image](https://github.com/user-attachments/assets/09c4327c-51aa-4ac3-867e-be879df9f8b9)


Output-



