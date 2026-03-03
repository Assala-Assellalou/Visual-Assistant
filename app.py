import gradio as gr
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
from PIL import Image
import torch
import tempfile
import os

# Load models
yolo_model = YOLO("yolov8n.pt")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Obstacles we care about for navigation
OBSTACLE_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "stop sign", "bench", "chair", "dining table",
    "stairs", "door", "cat", "dog", "suitcase", "backpack"
]

def analyze_scene(image):
    if image is None:
        return None, "No image provided", None

    # 1. YOLOv8 - Detect obstacles
    results = yolo_model(image)
    annotated_frame = results[0].plot()

    detected = []
    for box in results[0].boxes:
        class_name = yolo_model.names[int(box.cls)]
        confidence = float(box.conf)
        if class_name in OBSTACLE_CLASSES and confidence > 0.4:
            detected.append(f"{class_name} ({confidence:.0%})")

    # 2. BLIP - Generate scene description
    pil_image = Image.fromarray(image).convert("RGB")
    inputs = processor(pil_image, return_tensors="pt")
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_new_tokens=50)
    description = processor.decode(out[0], skip_special_tokens=True)

    # 3. Build audio message
    if detected:
        obstacle_text = ", ".join(detected[:3])
        audio_message = f"Warning. Obstacles detected: {obstacle_text}. Scene description: {description}"
    else:
        audio_message = f"Path appears clear. {description}"

    # 4. Text-to-Speech
    tts = gTTS(text=audio_message, lang="en")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)

    return annotated_frame, audio_message, tmp.name


# Gradio interface
demo = gr.Interface(
    fn=analyze_scene,
    inputs=gr.Image(label="Upload a scene"),
    outputs=[
        gr.Image(label="Detected obstacles"),
        gr.Textbox(label="Scene analysis"),
        gr.Audio(label="Audio guidance")
    ],
    title="Visual Assistant for the Visually Impaired",
    description="Upload an image to detect obstacles and get an audio description of the scene."
)

demo.launch()