# Visual Assistant — AI Navigation Aid for the Visually Impaired

A computer vision application that detects obstacles in real-time and generates audio scene descriptions to assist visually impaired users. Inspired by accessibility features in Meta Ray-Ban glasses and Google Lens.

---

## How it works

```
Image → YOLOv8 obstacle detection → BLIP scene description → gTTS audio guidance
```

1. Upload an image of any scene
2. YOLOv8 detects and localizes obstacles (people, stairs, vehicles, doors...)
3. BLIP generates a natural language description of the scene
4. gTTS converts the analysis into an audio guidance message

---

## Demo

| Input           | Output                                                                                                                        |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Street scene    | "Warning. Obstacles detected: person (92%), car (87%). Scene description: a busy street with people walking on the sidewalk." |
| Indoor corridor | "Path appears clear. A long hallway with doors on both sides."                                                                |

---

## Tech Stack

| Tool                 | Role                                          |
| -------------------- | --------------------------------------------- |
| YOLOv8 (Ultralytics) | Real-time obstacle detection and localization |
| BLIP (Salesforce)    | Image captioning and scene understanding      |
| gTTS                 | Text-to-speech audio guidance                 |
| Gradio               | Web interface                                 |
| PyTorch              | Deep learning inference                       |

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/Assala-Assellalou/Visual-Assistant.git
cd Visual-Assistant
```

**2. Install dependencies**

```bash
python -m pip install ultralytics transformers gradio gtts pillow torch torchvision
```

**3. Run the app**

```bash
python app.py
```

Open your browser at `http://127.0.0.1:7860`

> Note: BLIP model (~990MB) will be downloaded automatically on first launch.

---

## Project Structure

```
Visual-Assistant/
├── app.py          # Main application
├── yolov8n.pt      # YOLOv8 nano weights (auto-downloaded)
├── .gitignore
└── README.md
```

---

## Detected Obstacle Classes

Person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign, bench, chair, stairs, door, cat, dog, suitcase, backpack.

---

## Roadmap

* [ ] Real-time webcam support
* [ ] Turn-by-turn navigation guidance
* [ ] Distance estimation to obstacles
* [ ] Multi-language audio output
* [ ] Mobile deployment

---

## Motivation

Every year, millions of visually impaired people face challenges navigating unfamiliar environments. This project explores how computer vision and AI can provide real-time, affordable assistance — a concept at the core of accessibility initiatives at Meta and Google.
