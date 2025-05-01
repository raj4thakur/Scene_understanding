import torch
import cv2
import numpy as np
from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
from yolov5 import YOLOv5
import torchvision.transforms as T

# Load YOLOv5 model for object detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Function to load image from URL or local path
def load_image(image_path):
    if image_path.startswith('http'):
        image = Image.open(requests.get(image_path, stream=True).raw)
    else:
        image = Image.open(image_path)
    return image

# Function to perform object detection using YOLOv5
def detect_objects(image_path):
    # Load image
    img = cv2.imread(image_path) if not image_path.startswith('http') else np.array(load_image(image_path))

    # Perform inference
    results = model(img)

    # Parse results
    detected_objects = []
    for detection in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2, conf, cls = detection
        label = results.names[int(cls)]
        detected_objects.append({
            'label': label,
            'confidence': conf,
            'box': [x1, y1, x2, y2]
        })

    return detected_objects, img

# Function to generate image caption using BLIP
def generate_caption(image_path, max_length=20):
    # Load the BLIP model and processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Load and preprocess the image
    image = load_image(image_path)
    inputs = processor(image, return_tensors="pt")

    # Generate the caption
    outputs = model.generate(**inputs, max_length=max_length)
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption

# Function to generate scene graph (mock implementation)
def generate_scene_graph(detected_objects):
    relationships = []
    for i, obj1 in enumerate(detected_objects):
        for j, obj2 in enumerate(detected_objects):
            if i != j:
                # Mock relationship: "obj1 is near obj2"
                relationships.append({
                    'subject': obj1['label'],
                    'object': obj2['label'],
                    'relationship': 'near'
                })
    return relationships

# Function to combine object detection, image captioning, and scene graph generation for scene understanding
def scene_understanding(image_path):
    detected_objects, img = detect_objects(image_path)
    caption = generate_caption(image_path)
    scene_graph = generate_scene_graph(detected_objects)

    # Draw bounding boxes on the image
    for obj in detected_objects:
        x1, y1, x2, y2 = map(int, obj['box'])
        label = obj['label']
        confidence = obj['confidence']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Convert image to PIL format
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return detected_objects, caption, scene_graph, img_pil

# Main function
if __name__ == "__main__":
    image_path = 'E:\jupyter notebook\Computer Vision\Scene understanding\OIP.jpg'  # Replace with your image path or URL

    detected_objects, caption, scene_graph, annotated_image = scene_understanding(image_path)

    # Print results
    print("Detected Objects:")
    for obj in detected_objects:
        print(f"Label: {obj['label']}, Confidence: {obj['confidence']:.2f}, Box: {obj['box']}")

    print(f"Caption: {caption}")

    print("Scene Graph:")
    for rel in scene_graph:
        print(f"{rel['subject']} is {rel['relationship']} {rel['object']}")

    # Show annotated image
    annotated_image.show()
